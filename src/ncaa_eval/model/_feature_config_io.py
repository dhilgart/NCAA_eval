"""FeatureConfig serialization helpers shared by all model implementations."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ncaa_eval.transform.feature_serving import FeatureConfig


def save_feature_config(feature_config: FeatureConfig, path: Path) -> None:
    """Write ``feature_config.json`` sidecar file alongside model artifacts."""
    (path / "feature_config.json").write_text(json.dumps(asdict(feature_config), default=str))


def load_feature_config(path: Path) -> FeatureConfig | None:
    """Read ``feature_config.json`` sidecar, returning ``None`` if absent."""
    feature_config_path = path / "feature_config.json"
    if not feature_config_path.exists():
        return None
    data = json.loads(feature_config_path.read_text())
    return _deserialize_feature_config(data)


def _deserialize_feature_config(data: dict[str, Any]) -> FeatureConfig:
    """Reconstruct a FeatureConfig from a JSON-decoded dict."""
    from ncaa_eval.transform.elo import EloConfig

    elo_config_raw = data.pop("elo_config", None)
    elo_config = EloConfig(**elo_config_raw) if isinstance(elo_config_raw, dict) else None
    # Convert lists back to tuples for frozen dataclass compatibility
    for key in ("sequential_windows", "ewma_alphas", "batch_rating_types"):
        if key in data and isinstance(data[key], list):
            data[key] = tuple(data[key])
    if "ordinal_systems" in data and isinstance(data["ordinal_systems"], list):
        data["ordinal_systems"] = tuple(data["ordinal_systems"])
    return FeatureConfig(**data, elo_config=elo_config)
