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
from pathlib import Path
from typing import TYPE_CHECKING, Any

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


class StackedEnsembleConfig(ModelConfig):
    """Configuration record for a stacked ensemble.

    Stores base model types and contextual feature names for
    serialisation and run-tracking purposes.
    """

    model_name: str = "ensemble"
    base_model_types: list[str] = []  # noqa: RUF012
    contextual_features: list[str] = []  # noqa: RUF012


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

        # Load base models
        base_dir = path / "base_models"
        base_models: list[Model] = []
        for i in range(base_model_count):
            model_type = base_model_types[i]
            model_cls = get_model(model_type)
            base_models.append(model_cls.load(base_dir / f"base_{i}"))

        # Load meta-learner
        meta_type_path = path / "meta_learner" / "config.json"
        meta_config_data = json.loads(meta_type_path.read_text())
        meta_model_name = meta_config_data.get("model_name", "logistic_regression")
        meta_cls = get_model(meta_model_name)
        meta_learner = meta_cls.load(path / "meta_learner")

        return cls(
            base_models=base_models,
            meta_learner=meta_learner,
            contextual_features=contextual_features,
        )
