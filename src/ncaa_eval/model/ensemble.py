"""Stacked ensemble model — orchestrates base models and a meta-learner.

``StackedEnsemble`` is a standalone ``@dataclass`` (not a ``Model`` subclass)
that holds a list of base ``Model`` instances and a stateless meta-learner.
The training pipeline in ``cli/train.py`` dispatches on
``isinstance(model, StackedEnsemble)`` to invoke ensemble-specific training.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]
from pydantic import Field as _Field

from ncaa_eval.model._feature_config_io import save_feature_config
from ncaa_eval.model.base import Model, ModelConfig, StatefulModel
from ncaa_eval.model.registry import get_model, register_model
from ncaa_eval.transform.feature_serving import (
    DatasetScope,
    FeatureConfig,
    GenderScope,
    OrdinalCompositeMethod,
)

if TYPE_CHECKING:
    from ncaa_eval.transform.elo import EloConfig

logger = logging.getLogger(__name__)


# ── Feature-config union helpers (extracted for complexity budget) ──────────


def _resolve_elo(
    configs: list[FeatureConfig],
) -> tuple[bool, EloConfig | None]:
    """Return (elo_enabled, elo_config) from the union of *configs*."""
    elo_enabled = any(c.elo_enabled for c in configs)
    elo_config: EloConfig | None = None
    if elo_enabled:
        for c in configs:
            if c.elo_enabled and c.elo_config is not None:
                elo_config = c.elo_config
                break
    return elo_enabled, elo_config


def _resolve_ordinals(
    configs: list[FeatureConfig],
) -> tuple[OrdinalCompositeMethod | None, tuple[str, ...] | None]:
    """Return (ordinal_composite, ordinal_systems) from the union of *configs*."""
    ordinal_composite: OrdinalCompositeMethod | None = None
    for c in configs:
        if c.ordinal_composite is not None:
            ordinal_composite = c.ordinal_composite
            break

    systems: set[str] = set()
    any_systems = False
    for c in configs:
        if c.ordinal_systems is not None:
            any_systems = True
            systems.update(c.ordinal_systems)
    ordinal_systems: tuple[str, ...] | None = tuple(sorted(systems)) if any_systems else None
    return ordinal_composite, ordinal_systems


def _assert_agreement(
    configs: list[FeatureConfig],
) -> tuple[bool, GenderScope, DatasetScope]:
    """Assert matchup_deltas / gender_scope / dataset_scope agree across *configs*."""
    matchup_deltas_set = {c.matchup_deltas for c in configs}
    if len(matchup_deltas_set) > 1:
        msg = "All base models must agree on matchup_deltas"
        raise ValueError(msg)

    gender_scope_set = {c.gender_scope for c in configs}
    if len(gender_scope_set) > 1:
        msg = "All base models must agree on gender_scope"
        raise ValueError(msg)

    dataset_scope_set = {c.dataset_scope for c in configs}
    if len(dataset_scope_set) > 1:
        msg = "All base models must agree on dataset_scope"
        raise ValueError(msg)

    return matchup_deltas_set.pop(), gender_scope_set.pop(), dataset_scope_set.pop()


# ── Bracket prediction helpers (extracted for complexity budget) ─────────


def _discover_team_ids(data_dir: Path, season: int) -> list[int]:
    """Discover tournament-eligible team IDs from game data.

    Prefers tournament games (``is_tournament=True``) so the result is
    limited to the ~64 teams that actually competed in the bracket.  Falls
    back to all season games if no tournament games are found (e.g. for
    seasons before the tournament or when running on partial data).
    """
    from ncaa_eval.ingest import ParquetRepository

    repo = ParquetRepository(base_path=data_dir)
    games = repo.get_games(season)
    if not games:
        msg = f"No season data for bracket prediction: season {season}"
        raise FileNotFoundError(msg)

    tourney_games = [g for g in games if g.is_tournament]
    source_games = tourney_games if tourney_games else games

    team_id_set: set[int] = set()
    for g in source_games:
        team_id_set.add(g.w_team_id)
        team_id_set.add(g.l_team_id)
    return sorted(team_id_set)


def _stateful_base_matrix(
    model: StatefulModel,
    team_ids: list[int],
    season: int,
) -> npt.NDArray[np.float64]:
    """Build pairwise probability matrix for a stateful base model."""
    from ncaa_eval.evaluation.bracket import MatchupContext
    from ncaa_eval.evaluation.kaggle_export import KAGGLE_NEUTRAL_DAY_NUM
    from ncaa_eval.evaluation.providers import EloProvider, build_probability_matrix

    provider = EloProvider(model)
    context = MatchupContext(
        season=season,
        day_num=KAGGLE_NEUTRAL_DAY_NUM,
        is_neutral=True,
    )
    return build_probability_matrix(provider, team_ids, context)


def _extract_team_features(
    season_df: pd.DataFrame,
    team_ids: list[int],
) -> dict[int, dict[str, float]]:
    """Extract per-team feature profiles from season game data.

    For each team, averages their features across all games where they
    appear as either team_a or team_b.  Returns a mapping of
    ``team_id -> {feature_name_without_suffix: value}``.
    """
    # Identify per-team columns (those ending with _a or _b)
    a_cols = [c for c in season_df.columns if c.endswith("_a") and c != "team_a_id"]
    b_cols = [c for c in season_df.columns if c.endswith("_b") and c != "team_b_id"]
    # Map base feature name to (a_col, b_col)
    base_names: dict[str, tuple[str, str]] = {}
    for ac in a_cols:
        base = ac[:-2]  # strip _a
        bc = f"{base}_b"
        if bc in b_cols:
            base_names[base] = (ac, bc)

    team_profiles: dict[int, dict[str, float]] = {}
    for tid in team_ids:
        # Games where this team is team_a
        as_a = season_df[season_df["team_a_id"] == tid]
        # Games where this team is team_b
        as_b = season_df[season_df["team_b_id"] == tid]

        profile: dict[str, float] = {}
        for base, (ac, bc) in base_names.items():
            values: list[float] = []
            if not as_a.empty and ac in as_a.columns:
                vals = as_a[ac].dropna()
                values.extend(vals.tolist())
            if not as_b.empty and bc in as_b.columns:
                vals = as_b[bc].dropna()
                values.extend(vals.tolist())
            profile[base] = float(np.mean(values)) if values else 0.0

        team_profiles[tid] = profile
    return team_profiles


def _build_synthetic_feature_rows(
    team_profiles: dict[int, dict[str, float]],
    team_ids: list[int],
    feat_names: list[str],
) -> pd.DataFrame:
    """Build synthetic feature rows for all C(n,2) team pairings.

    Constructs ``_a``/``_b`` columns from team profiles and computes
    ``delta_*`` and ``seed_diff`` columns.
    """
    rows: list[dict[str, float]] = []
    pair_indices: list[tuple[int, int]] = []
    for i, j in combinations(range(len(team_ids)), 2):
        tid_a, tid_b = team_ids[i], team_ids[j]
        prof_a = team_profiles.get(tid_a, {})
        prof_b = team_profiles.get(tid_b, {})
        row: dict[str, float] = {}
        # Populate _a and _b columns (dict.fromkeys preserves insertion order,
        # deduplicating without non-deterministic set iteration)
        for base in dict.fromkeys(list(prof_a.keys()) + list(prof_b.keys())):
            row[f"{base}_a"] = prof_a.get(base, 0.0)
            row[f"{base}_b"] = prof_b.get(base, 0.0)
            row[f"delta_{base}"] = row[f"{base}_a"] - row[f"{base}_b"]
        # Special: seed_diff (derived from seed_num)
        if "seed_num" in prof_a or "seed_num" in prof_b:
            row["seed_diff"] = prof_a.get("seed_num", 0.0) - prof_b.get("seed_num", 0.0)
        rows.append(row)
        pair_indices.append((i, j))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Fill missing feature columns with 0
    for c in feat_names:
        if c not in df.columns:
            df[c] = 0.0
    df.attrs["pair_indices"] = pair_indices
    return df[feat_names]


def _stateless_base_matrix(
    model: Model,
    team_ids: list[int],
    data_dir: Path,
    season: int,
) -> npt.NDArray[np.float64]:
    """Build pairwise probability matrix for a stateless base model."""
    from ncaa_eval.cli.train import _setup_feature_server

    feat_names: list[str] = model.feature_names_  # type: ignore[attr-defined]
    server = _setup_feature_server(data_dir, model.feature_config)
    season_df = server.serve_season_features(season, mode="batch")

    team_profiles = _extract_team_features(season_df, team_ids)
    synthetic_df = _build_synthetic_feature_rows(team_profiles, team_ids, feat_names)

    n = len(team_ids)
    P = np.zeros((n, n), dtype=np.float64)

    if synthetic_df.empty:
        return P

    pair_indices: list[tuple[int, int]] = synthetic_df.attrs["pair_indices"]
    probs = model.predict_proba(synthetic_df)

    for idx, (i, j) in enumerate(pair_indices):
        p = float(probs.iloc[idx])
        P[i, j] = p
        P[j, i] = 1.0 - p

    return P


def _predict_bracket_base_matrices(
    base_models: list[Model],
    data_dir: Path,
    season: int,
) -> tuple[list[int], list[npt.NDArray[np.float64]]]:
    """Build per-base-model probability matrices.

    Returns:
        Tuple of (team_ids, list of n×n probability matrices).
    """
    team_ids = _discover_team_ids(data_dir, season)
    matrices: list[npt.NDArray[np.float64]] = []

    for base_model in base_models:
        if isinstance(base_model, StatefulModel):
            mat = _stateful_base_matrix(base_model, team_ids, season)
        else:
            mat = _stateless_base_matrix(base_model, team_ids, data_dir, season)
        matrices.append(mat)

    return team_ids, matrices


def _build_bracket_contextual_features(
    data_dir: Path,
    season: int,
    team_ids: list[int],
    contextual_features: list[str],
) -> dict[str, npt.NDArray[np.float64]]:
    """Build contextual feature vectors for all C(n,2) matchups.

    Returns a dict mapping feature name to a 1-D array of length C(n,2),
    ordered by upper-triangle iteration.
    """
    from ncaa_eval.cli.train import _setup_feature_server

    if not contextual_features:
        return {}

    n = len(team_ids)

    # Build a feature server to get contextual features
    server = _setup_feature_server(data_dir, FeatureConfig())
    season_df = server.serve_season_features(season, mode="batch")

    # Extract per-team contextual profiles
    team_profiles = _extract_team_features(season_df, team_ids)

    result: dict[str, npt.NDArray[np.float64]] = {}
    for feat in contextual_features:
        values: list[float] = []
        for i, j in combinations(range(n), 2):
            tid_a, tid_b = team_ids[i], team_ids[j]
            prof_a = team_profiles.get(tid_a, {})
            prof_b = team_profiles.get(tid_b, {})

            if feat == "seed_diff":
                val = prof_a.get("seed_num", 0.0) - prof_b.get("seed_num", 0.0)
            elif feat == "is_tournament":
                val = 1.0  # bracket prediction is always tournament
            elif feat == "loc_encoding":
                val = 0.0  # neutral site for bracket
            else:
                # Generic: use team A's value minus team B's value if available
                val = prof_a.get(feat, 0.0) - prof_b.get(feat, 0.0)
            values.append(val)
        result[feat] = np.array(values, dtype=np.float64)

    return result


def _assemble_bracket_meta_predictions(
    *,
    base_matrices: list[npt.NDArray[np.float64]],
    context_features: dict[str, npt.NDArray[np.float64]],
    meta_column_order: list[str],
    meta_learner: Model,
    n: int,
) -> npt.NDArray[np.float64]:
    """Assemble meta-input from base matrices + context and predict.

    Returns the final n×n probability matrix.
    """
    # Extract upper-triangle predictions from each base matrix
    rows_idx, cols_idx = np.triu_indices(n, k=1)
    expected_len = len(rows_idx)  # C(n, 2)
    meta_parts: dict[str, npt.NDArray[np.float64]] = {}
    for i, mat in enumerate(base_matrices):
        meta_parts[f"pred_base_{i}"] = mat[rows_idx, cols_idx].astype(np.float64)

    # Add contextual features (validate length matches upper-triangle count)
    for feat, vals in context_features.items():
        if len(vals) != expected_len:
            msg = f"Context feature {feat!r} has length {len(vals)}, expected {expected_len} (C({n}, 2))"
            raise ValueError(msg)
        meta_parts[feat] = vals

    meta_df = pd.DataFrame(meta_parts)

    # Validate column order
    missing = [c for c in meta_column_order if c not in meta_df.columns]
    if missing:
        msg = f"Missing meta-learner input columns: {missing}"
        raise ValueError(msg)

    meta_X = meta_df[meta_column_order]
    probs = meta_learner.predict_proba(meta_X)

    # Fill n×n matrix
    P = np.zeros((n, n), dtype=np.float64)
    P[rows_idx, cols_idx] = probs.values.astype(np.float64)
    P[cols_idx, rows_idx] = 1.0 - probs.values.astype(np.float64)
    return P


class StackedEnsembleConfig(ModelConfig):
    """Configuration record for a stacked ensemble.

    Stores base model types and contextual feature names for
    serialisation and run-tracking purposes.
    """

    model_name: str = "ensemble"
    base_model_types: list[str] = _Field(default_factory=list)
    contextual_features: list[str] = _Field(default_factory=list)


# Register so that ``list_models()`` includes ``"ensemble"``
# and ``RunStore.load_model`` can resolve ``model_type="ensemble"``.
# We register a sentinel — the real "load" path is via
# ``StackedEnsemble.load()``, not ``EnsembleSentinel.load()``.
@register_model("ensemble")
class _EnsembleSentinel(Model):
    """Registry placeholder — never instantiated directly."""

    feature_config = FeatureConfig()

    def fit(self, X: Any, y: Any) -> None:  # pragma: no cover
        raise NotImplementedError

    def predict_proba(self, X: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def save(self, path: Path) -> None:  # pragma: no cover
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path) -> _EnsembleSentinel:  # pragma: no cover
        raise NotImplementedError

    def get_config(self) -> ModelConfig:  # pragma: no cover
        raise NotImplementedError


@dataclass
class StackedEnsemble:
    """Stacked generalisation ensemble.

    Holds a list of base ``Model`` instances and a stateless meta-learner.
    The ensemble's ``feature_config`` is the union of all base models' configs.

    Attributes:
        base_models: Two or more trained (or to-be-trained) base models.
        meta_learner: A stateless ``Model`` that learns to combine base
            model predictions with contextual features.
        contextual_features: Column names appended to OOF predictions
            before meta-learner training (e.g. ``seed_diff``).
    """

    base_models: list[Model]
    meta_learner: Model
    contextual_features: list[str] = field(
        default_factory=lambda: ["seed_diff", "is_tournament", "loc_encoding"],
    )
    meta_column_order: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate base model count and meta-learner type."""
        if isinstance(self.meta_learner, StatefulModel):
            msg = "meta_learner must be a stateless Model, not StatefulModel"
            raise TypeError(msg)
        if len(self.base_models) < 2:
            msg = "StackedEnsemble requires at least 2 base models"
            raise ValueError(msg)

    # ------------------------------------------------------------------
    # feature_config — union of all base models
    # ------------------------------------------------------------------

    @property
    def feature_config(self) -> FeatureConfig:
        """Return the union of all base model feature configs."""
        configs = [m.feature_config for m in self.base_models]

        elo_enabled, elo_config = _resolve_elo(configs)
        ordinal_composite, ordinal_systems = _resolve_ordinals(configs)
        matchup_deltas, gender_scope, dataset_scope = _assert_agreement(configs)

        return FeatureConfig(
            sequential_windows=tuple(sorted({w for c in configs for w in c.sequential_windows})),
            ewma_alphas=tuple(sorted({a for c in configs for a in c.ewma_alphas})),
            graph_features_enabled=any(c.graph_features_enabled for c in configs),
            batch_rating_types=tuple(sorted({t for c in configs for t in c.batch_rating_types})),
            ordinal_systems=ordinal_systems,
            ordinal_composite=ordinal_composite,
            matchup_deltas=matchup_deltas,
            gender_scope=gender_scope,
            dataset_scope=dataset_scope,
            elo_enabled=elo_enabled,
            elo_config=elo_config,
        )

    # ------------------------------------------------------------------
    # Pydantic config helper (for run tracking)
    # ------------------------------------------------------------------

    def get_config(self) -> StackedEnsembleConfig:
        """Return a serialisable configuration record."""
        base_types: list[str] = []
        for m in self.base_models:
            cfg = m.get_config()
            base_types.append(cfg.model_name)
        return StackedEnsembleConfig(
            base_model_types=base_types,
            contextual_features=list(self.contextual_features),
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Route features through base models and meta-learner.

        For each base model, generates predictions by dispatching
        stateless models through ``X[base_model.feature_names_]`` and
        stateful models through the full ``X``.  Assembles base
        predictions and contextual features into a meta-input DataFrame
        in ``self.meta_column_order``, then calls the meta-learner.

        Args:
            X: Feature DataFrame with at least the columns required by
                each base model and all contextual features.

        Returns:
            Series of ensemble win probabilities, indexed like *X*.

        Raises:
            ValueError: If any column in ``meta_column_order`` is missing
                from the assembled meta-input.
        """
        base_preds: dict[str, pd.Series[float]] = {}
        for i, base_model in enumerate(self.base_models):
            col_name = f"pred_base_{i}"
            if isinstance(base_model, StatefulModel):
                base_preds[col_name] = base_model.predict_proba(X)
            else:
                feat_names: list[str] = base_model.feature_names_  # type: ignore[attr-defined]
                base_preds[col_name] = base_model.predict_proba(X[feat_names])

        # Assemble meta-input in training column order
        meta_parts: dict[str, pd.Series[float]] = {**base_preds}
        for feat in self.contextual_features:
            if feat in X.columns:
                meta_parts[feat] = X[feat]

        meta_df = pd.DataFrame(meta_parts, index=X.index)

        # Validate column order
        missing = [c for c in self.meta_column_order if c not in meta_df.columns]
        if missing:
            msg = f"Missing meta-learner input columns: {missing}"
            raise ValueError(msg)

        meta_X = meta_df[self.meta_column_order]
        return self.meta_learner.predict_proba(meta_X)

    def predict_bracket(
        self,
        data_dir: Path,
        season: int,
    ) -> pd.DataFrame:
        """Generate an n×n pairwise probability matrix for bracket prediction.

        Discovers tournament-eligible teams, generates per-base-model
        pairwise predictions, assembles meta-input for all C(n,2) matchups,
        and returns a probability matrix suitable for the Monte Carlo
        bracket simulator.

        Args:
            data_dir: Path to the local Parquet data store.
            season: Target season year.

        Returns:
            DataFrame of shape ``(n, n)`` with team_id index and columns.
            ``P[a, b]`` is the ensemble probability that team *a* beats
            team *b*.  Diagonal is zero; ``P[a,b] + P[b,a] ≈ 1``.

        Raises:
            FileNotFoundError: If no season data exists for *season*.
            ValueError: If any column in ``meta_column_order`` is missing
                from the assembled meta-input, or if a context feature
                array has unexpected length.
        """
        team_ids, base_matrices = _predict_bracket_base_matrices(self.base_models, data_dir, season)
        n = len(team_ids)

        # Build contextual feature vectors for all C(n,2) pairs
        context_features = _build_bracket_contextual_features(
            data_dir, season, team_ids, self.contextual_features
        )

        # Assemble meta-input for upper triangle and predict
        prob_matrix = _assemble_bracket_meta_predictions(
            base_matrices=base_matrices,
            context_features=context_features,
            meta_column_order=self.meta_column_order,
            meta_learner=self.meta_learner,
            n=n,
        )

        return pd.DataFrame(prob_matrix, index=team_ids, columns=team_ids)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save the ensemble to *path*.

        Layout::

            path/
              manifest.json
              feature_config.json
              base_models/
                base_0/  …  (Model.save)
                base_1/  …
              meta_learner/  …  (Model.save)
        """
        path.mkdir(parents=True, exist_ok=True)

        # Base models
        base_dir = path / "base_models"
        base_dir.mkdir(exist_ok=True)
        base_model_types: list[str] = []
        for i, model in enumerate(self.base_models):
            model_path = base_dir / f"base_{i}"
            model.save(model_path)
            base_model_types.append(model.get_config().model_name)

        # Meta-learner
        meta_path = path / "meta_learner"
        self.meta_learner.save(meta_path)

        # Manifest
        manifest: dict[str, Any] = {
            "base_model_types": base_model_types,
            "base_model_count": len(self.base_models),
            "contextual_features": list(self.contextual_features),
            "meta_learner_type": self.meta_learner.get_config().model_name,
            "meta_column_order": list(self.meta_column_order),
        }
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2))

        # Feature config
        save_feature_config(self.feature_config, path)

    @classmethod
    def load(cls, path: Path) -> StackedEnsemble:
        """Reconstruct a ``StackedEnsemble`` from a saved directory."""
        manifest_data = json.loads((path / "manifest.json").read_text())
        base_model_types: list[str] = manifest_data["base_model_types"]
        base_model_count: int = manifest_data["base_model_count"]
        contextual_features: list[str] = manifest_data["contextual_features"]
        meta_learner_type: str = manifest_data["meta_learner_type"]
        meta_column_order: list[str] = manifest_data.get("meta_column_order", [])

        # Load base models
        base_dir = path / "base_models"
        base_models: list[Model] = []
        for i in range(base_model_count):
            model_type = base_model_types[i]
            model_cls = get_model(model_type)
            base_models.append(model_cls.load(base_dir / f"base_{i}"))

        # Load meta-learner
        meta_cls = get_model(meta_learner_type)
        meta_learner = meta_cls.load(path / "meta_learner")

        return cls(
            base_models=base_models,
            meta_learner=meta_learner,
            contextual_features=contextual_features,
            meta_column_order=meta_column_order,
        )
