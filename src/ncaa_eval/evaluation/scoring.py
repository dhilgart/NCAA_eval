"""Scoring rule protocols, implementations, and registry.

Provides the :class:`ScoringRule` protocol, concrete scoring
implementations, and a decorator-based registry:

* :class:`ScoringRule` — protocol for tournament bracket scoring rules.
* :class:`StandardScoring` — ESPN-style 1-2-4-8-16-32.
* :class:`FibonacciScoring` — 2-3-5-8-13-21.
* :class:`SeedDiffBonusScoring` — base + seed-difference upset bonus.
* :class:`CustomScoring` — user-defined callable-based scoring.
* :class:`DictScoring` — dict-based scoring.
* :func:`register_scoring` — class decorator for registry registration.
* :func:`get_scoring` — retrieve a scoring class by name.
* :func:`list_scorings` — list all registered scoring names.
* :func:`scoring_from_config` — create a scoring rule from a config dict.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Protocol, TypeVar, runtime_checkable

from ncaa_eval.evaluation.bracket import N_ROUNDS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scoring rule protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ScoringRule(Protocol):
    """Protocol for tournament bracket scoring rules."""

    @property
    def name(self) -> str:
        """Human-readable name of the scoring rule."""
        ...

    def points_per_round(self, round_idx: int) -> float:
        """Return points awarded for a correct pick in round *round_idx*.

        Args:
            round_idx: Zero-indexed round number (0=R64 through 5=NCG).

        Returns:
            Points as a float.
        """
        ...


# ---------------------------------------------------------------------------
# Scoring registry (decorator-based, mirrors model/registry.py)
# ---------------------------------------------------------------------------

_ST = TypeVar("_ST", bound="type[ScoringRule]")

_SCORING_REGISTRY: dict[str, type[ScoringRule]] = {}
_SCORING_DISPLAY_NAMES: dict[str, str] = {}


class ScoringNotFoundError(KeyError):
    """Raised when a requested scoring name is not in the registry."""


def register_scoring(name: str, *, display_name: str | None = None) -> Callable[[_ST], _ST]:
    """Class decorator that registers a scoring rule class.

    Args:
        name: Registry key for the scoring rule.
        display_name: Optional human-readable label for UI display.
            Falls back to *name* if not provided.

    Returns:
        Decorator that registers the class and returns it unchanged.

    Raises:
        ValueError: If *name* is already registered.
    """

    def decorator(cls: _ST) -> _ST:
        if name in _SCORING_REGISTRY:
            msg = f"Scoring name {name!r} is already registered to {_SCORING_REGISTRY[name].__name__}"
            raise ValueError(msg)
        _SCORING_REGISTRY[name] = cls
        _SCORING_DISPLAY_NAMES[name] = display_name or name
        return cls

    return decorator


def get_scoring(name: str) -> type:
    """Return the scoring class registered under *name*.

    Raises:
        ScoringNotFoundError: If *name* is not registered.
    """
    try:
        return _SCORING_REGISTRY[name]
    except KeyError:
        msg = f"No scoring registered with name {name!r}. Available: {list_scorings()}"
        raise ScoringNotFoundError(msg) from None


def list_scorings() -> list[str]:
    """Return all registered scoring names (sorted)."""
    return sorted(_SCORING_REGISTRY)


def list_scoring_display_names() -> dict[str, str]:
    """Return a mapping of registry keys to display names.

    Returns:
        Dict mapping scoring name → human-readable display name.
    """
    return dict(sorted(_SCORING_DISPLAY_NAMES.items()))


# ---------------------------------------------------------------------------
# Scoring rule implementations
# ---------------------------------------------------------------------------


@register_scoring("standard", display_name="Standard (1-2-4-8-16-32)")
class StandardScoring:
    """ESPN-style scoring: 1-2-4-8-16-32 (192 total for perfect bracket)."""

    _POINTS: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)

    @property
    def name(self) -> str:
        """Return ``'standard'``."""
        return "standard"

    def points_per_round(self, round_idx: int) -> float:
        """Return standard scoring points for *round_idx*."""
        return self._POINTS[round_idx]


@register_scoring("fibonacci", display_name="Fibonacci (2-3-5-8-13-21)")
class FibonacciScoring:
    """Fibonacci-style scoring: 2-3-5-8-13-21 (231 total for perfect bracket)."""

    _POINTS: tuple[float, ...] = (2.0, 3.0, 5.0, 8.0, 13.0, 21.0)

    @property
    def name(self) -> str:
        """Return ``'fibonacci'``."""
        return "fibonacci"

    def points_per_round(self, round_idx: int) -> float:
        """Return Fibonacci scoring points for *round_idx*."""
        return self._POINTS[round_idx]


@register_scoring("seed_diff_bonus")
class SeedDiffBonusScoring:
    """Base points + seed-difference bonus when lower seed wins.

    Uses same base as StandardScoring (1-2-4-8-16-32).  When the lower
    seed (higher seed number) wins, adds ``|seed_a - seed_b|`` bonus.

    Note: This scoring rule's ``points_per_round`` returns only the base
    points.  Full EP computation for seed-diff scoring (which requires
    per-matchup seed information) is deferred to Story 6.6, which will add
    a dedicated ``compute_expected_points_seed_diff`` function.

    Args:
        seed_map: Mapping of ``team_id → seed_num``.
    """

    _BASE_POINTS: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)

    def __init__(self, seed_map: dict[int, int]) -> None:
        self._seed_map = seed_map

    @property
    def name(self) -> str:
        """Return ``'seed_diff_bonus'``."""
        return "seed_diff_bonus"

    def points_per_round(self, round_idx: int) -> float:
        """Return base points (excludes seed-diff bonus)."""
        return self._BASE_POINTS[round_idx]

    def seed_diff_bonus(self, seed_a: int, seed_b: int) -> float:
        """Return bonus points when the lower seed wins.

        Args:
            seed_a: Winner's seed number.
            seed_b: Loser's seed number.

        Returns:
            ``|seed_a - seed_b|`` if winner has higher seed number
            (lower seed = upset), else 0.
        """
        if seed_a > seed_b:
            return float(abs(seed_a - seed_b))
        return 0.0

    @property
    def seed_map(self) -> dict[int, int]:
        """Return the seed lookup map."""
        return self._seed_map


class CustomScoring:
    """User-defined scoring rule wrapping a callable.

    Args:
        scoring_fn: Callable mapping ``round_idx`` → points.
        scoring_name: Name for this custom rule.
    """

    def __init__(self, scoring_fn: Callable[[int], float], scoring_name: str) -> None:
        self._fn = scoring_fn
        self._name = scoring_name

    @property
    def name(self) -> str:
        """Return the custom rule name."""
        return self._name

    def points_per_round(self, round_idx: int) -> float:
        """Return points from the wrapped callable."""
        return self._fn(round_idx)


class DictScoring:
    """Scoring rule from a dict mapping round_idx to points.

    Args:
        points: Mapping of ``round_idx → points`` for rounds 0–5.
        scoring_name: Name for this rule.

    Raises:
        ValueError: If *points* does not contain exactly 6 entries (rounds 0–5).
    """

    def __init__(self, points: dict[int, float], scoring_name: str) -> None:
        if len(points) != N_ROUNDS:
            msg = f"DictScoring requires exactly 6 entries (rounds 0–5), got {len(points)}"
            raise ValueError(msg)
        if set(points) != set(range(N_ROUNDS)):
            msg = f"DictScoring requires keys 0–5, got {sorted(points.keys())}"
            raise ValueError(msg)
        self._points = points
        self._name = scoring_name

    @property
    def name(self) -> str:
        """Return the rule name."""
        return self._name

    def points_per_round(self, round_idx: int) -> float:
        """Return points for *round_idx*."""
        return self._points[round_idx]


def scoring_from_config(config: dict[str, Any]) -> ScoringRule:
    """Create a scoring rule from a configuration dict.

    Dispatches on ``config["type"]``:

    * ``"standard"`` → :class:`StandardScoring`
    * ``"fibonacci"`` → :class:`FibonacciScoring`
    * ``"seed_diff_bonus"`` → :class:`SeedDiffBonusScoring` (requires ``seed_map``)
    * ``"dict"`` → :class:`DictScoring` (requires ``points`` and ``name``)
    * ``"custom"`` → :class:`CustomScoring` (requires ``callable`` and ``name``)

    Args:
        config: Configuration dict with at least a ``"type"`` key.

    Returns:
        Instantiated scoring rule.

    Raises:
        ValueError: If ``type`` is unknown or required keys are missing.
    """
    if "type" not in config:
        msg = "scoring config must contain a 'type' key"
        raise ValueError(msg)
    scoring_type = config["type"]
    if scoring_type == "standard":
        return StandardScoring()
    if scoring_type == "fibonacci":
        return FibonacciScoring()
    if scoring_type == "seed_diff_bonus":
        if "seed_map" not in config:
            msg = "scoring config for 'seed_diff_bonus' requires a 'seed_map' key"
            raise ValueError(msg)
        return SeedDiffBonusScoring(config["seed_map"])
    if scoring_type == "dict":
        if "points" not in config:
            msg = "scoring config for 'dict' requires a 'points' key"
            raise ValueError(msg)
        return DictScoring(config["points"], config.get("name", "dict"))
    if scoring_type == "custom":
        if "callable" not in config:
            msg = "scoring config for 'custom' requires a 'callable' key"
            raise ValueError(msg)
        return CustomScoring(config["callable"], config.get("name", "custom"))
    msg = f"Unknown scoring type: {scoring_type!r}"
    raise ValueError(msg)
