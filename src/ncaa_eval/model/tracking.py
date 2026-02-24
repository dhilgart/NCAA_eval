"""Model run tracking: metadata, predictions, and persistence.

Defines ``ModelRun`` and ``Prediction`` Pydantic records for run metadata
and game-level predictions, plus ``RunStore`` for local JSON/Parquet
persistence under ``base_path / "runs" / run_id /``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

from ncaa_eval.model.base import Model

# ── PyArrow schema for Prediction Parquet files ────────────────────────────

_PREDICTION_SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("game_id", pa.string()),
        ("season", pa.int64()),
        ("team_a_id", pa.int64()),
        ("team_b_id", pa.int64()),
        ("pred_win_prob", pa.float64()),
    ]
)


# ── Pydantic data entities ─────────────────────────────────────────────────


class ModelRun(BaseModel):
    """Metadata for a single model training run."""

    run_id: str
    model_type: str
    hyperparameters: dict[str, Any]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    git_hash: str
    start_year: int
    end_year: int
    prediction_count: int


class Prediction(BaseModel):
    """A single game-level probability prediction."""

    run_id: str
    game_id: str
    season: int
    team_a_id: int
    team_b_id: int
    pred_win_prob: Annotated[float, Field(ge=0.0, le=1.0)]


# ── Persistence layer ──────────────────────────────────────────────────────


class RunStore:
    """Persist and load model runs and predictions on the local filesystem.

    Directory layout::

        base_path/
          runs/
            <run_id>/
              run.json                    # ModelRun metadata
              predictions.parquet         # Prediction records (PyArrow)
              summary.parquet             # BacktestResult.summary (year × metrics)
              fold_predictions.parquet    # CV fold y_true/y_prob per year
              model/                      # Trained model artifacts
                model.ubj                 # XGBoost native format (XGBoost only)
                model.json                # Elo ratings (Elo only)
                config.json               # Model config
                feature_names.json        # Feature column names used during training
    """

    def __init__(self, base_path: Path) -> None:
        self._runs_dir = base_path / "runs"

    def save_run(self, run: ModelRun, predictions: list[Prediction]) -> None:
        """Write run metadata (JSON) and predictions (Parquet)."""
        run_dir = self._runs_dir / run.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Metadata
        (run_dir / "run.json").write_text(run.model_dump_json(indent=2))

        # Predictions
        if predictions:
            rows = [p.model_dump() for p in predictions]
            table = pa.Table.from_pylist(rows, schema=_PREDICTION_SCHEMA)
        else:
            table = pa.table(
                {
                    col: pa.array([], type=typ)
                    for col, typ in zip(
                        _PREDICTION_SCHEMA.names,
                        [f.type for f in _PREDICTION_SCHEMA],
                    )
                },
                schema=_PREDICTION_SCHEMA,
            )
        pq.write_table(table, run_dir / "predictions.parquet")

    def load_run(self, run_id: str) -> ModelRun:
        """Load run metadata from JSON.

        Raises
        ------
        FileNotFoundError
            If the run directory or ``run.json`` does not exist.
        """
        run_json = self._runs_dir / run_id / "run.json"
        if not run_json.exists():
            msg = f"No run found with id {run_id!r} at {run_json}"
            raise FileNotFoundError(msg)
        return ModelRun.model_validate_json(run_json.read_text())

    def load_predictions(self, run_id: str) -> pd.DataFrame:
        """Load predictions from Parquet as a DataFrame.

        Raises
        ------
        FileNotFoundError
            If the predictions Parquet file does not exist.
        """
        pq_path = self._runs_dir / run_id / "predictions.parquet"
        if not pq_path.exists():
            msg = f"No predictions found for run {run_id!r} at {pq_path}"
            raise FileNotFoundError(msg)
        return pq.read_table(pq_path).to_pandas()

    def save_metrics(self, run_id: str, summary: pd.DataFrame) -> None:
        """Persist backtest metric summary for a run.

        Args:
            run_id: The run identifier.
            summary: BacktestResult.summary DataFrame (index=year,
                columns=[log_loss, brier_score, roc_auc, ece, elapsed_seconds]).

        Raises:
            FileNotFoundError: If the run directory does not exist.
        """
        run_dir = self._runs_dir / run_id
        if not run_dir.exists():
            msg = f"Run directory not found: {run_id}"
            raise FileNotFoundError(msg)
        summary.to_parquet(run_dir / "summary.parquet")

    def load_metrics(self, run_id: str) -> pd.DataFrame | None:
        """Load backtest metric summary for a run.

        Args:
            run_id: The run identifier.

        Returns:
            Summary DataFrame or None if no summary exists (legacy run).
        """
        path = self._runs_dir / run_id / "summary.parquet"
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def save_fold_predictions(self, run_id: str, fold_preds: pd.DataFrame) -> None:
        """Persist fold-level predictions from walk-forward CV.

        Args:
            run_id: The run identifier.
            fold_preds: DataFrame with columns [year, game_id, team_a_id,
                team_b_id, pred_win_prob, team_a_won].

        Raises:
            FileNotFoundError: If the run directory does not exist.
        """
        run_dir = self._runs_dir / run_id
        if not run_dir.exists():
            msg = f"Run directory not found: {run_id}"
            raise FileNotFoundError(msg)
        fold_preds.to_parquet(run_dir / "fold_predictions.parquet", index=False)

    def load_fold_predictions(self, run_id: str) -> pd.DataFrame | None:
        """Load fold-level predictions for a run.

        Args:
            run_id: The run identifier.

        Returns:
            DataFrame or None if no fold predictions exist (legacy run).
        """
        path = self._runs_dir / run_id / "fold_predictions.parquet"
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def save_model(
        self,
        run_id: str,
        model: Model,
        *,
        feature_names: list[str] | None = None,
    ) -> None:
        """Persist a trained model alongside a run.

        Args:
            run_id: The run identifier.
            model: A fitted model implementing ``save(path)``.
            feature_names: Feature column names used during training.

        Raises:
            FileNotFoundError: If the run directory does not exist.
        """
        run_dir = self._runs_dir / run_id
        if not run_dir.exists():
            msg = f"Run directory not found: {run_id}"
            raise FileNotFoundError(msg)
        model_dir = run_dir / "model"
        model_dir.mkdir(exist_ok=True)
        model.save(model_dir)
        if feature_names is not None:
            (model_dir / "feature_names.json").write_text(json.dumps(feature_names))

    def load_model(self, run_id: str) -> Model | None:
        """Load a trained model from a run directory.

        Args:
            run_id: The run identifier.

        Returns:
            Model instance or None if no model directory exists (legacy run).
        """
        from ncaa_eval.model.registry import get_model

        model_dir = self._runs_dir / run_id / "model"
        if not model_dir.exists():
            return None
        run = self.load_run(run_id)
        model_cls = get_model(run.model_type)
        return model_cls.load(model_dir)

    def load_feature_names(self, run_id: str) -> list[str] | None:
        """Load saved feature names for a run.

        Args:
            run_id: The run identifier.

        Returns:
            List of feature names or None if not saved.
        """
        path = self._runs_dir / run_id / "model" / "feature_names.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())  # type: ignore[no-any-return]

    def load_all_summaries(self) -> pd.DataFrame:
        """Load metric summaries for all runs that have them.

        Returns:
            DataFrame with columns [run_id, year, log_loss, brier_score,
            roc_auc, ece, elapsed_seconds]. Empty DataFrame if no summaries.
        """
        frames: list[pd.DataFrame] = []
        for run in self.list_runs():
            summary = self.load_metrics(run.run_id)
            if summary is not None:
                df = summary.reset_index()
                df["run_id"] = run.run_id
                frames.append(df)
        if not frames:
            return pd.DataFrame(
                columns=["run_id", "year", "log_loss", "brier_score", "roc_auc", "ece", "elapsed_seconds"]
            )
        return pd.concat(frames, ignore_index=True)

    def list_runs(self) -> list[ModelRun]:
        """Scan the runs directory and return all saved ModelRun records."""
        if not self._runs_dir.exists():
            return []
        runs: list[ModelRun] = []
        for run_dir in sorted(self._runs_dir.iterdir()):
            run_json = run_dir / "run.json"
            if run_json.exists():
                runs.append(ModelRun.model_validate_json(run_json.read_text()))
        return runs
