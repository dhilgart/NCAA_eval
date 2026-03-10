"""Post-sync data validation checks.

Validates data quality after sync completes — game counts, duplicates,
and team reference integrity.  All checks are non-fatal: they produce
a :class:`ValidationReport` with pass/fail status per check rather than
raising exceptions.
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

from pydantic import BaseModel, ConfigDict

from ncaa_eval.ingest.repository import Repository
from ncaa_eval.utils.logger import get_logger

log = get_logger("ingest.validation")

# Season 2020 had a shortened season due to COVID-19 (no tournament).
_COVID_SEASON: int = 2020

# ±10% threshold for game count validation.
_GAME_COUNT_TOLERANCE: float = 0.10


class ValidationResult(BaseModel):
    """Result of a single validation check."""

    model_config = ConfigDict(frozen=True)

    check_name: str
    passed: bool
    message: str
    details: dict[str, Any] = {}


class ValidationReport(BaseModel):
    """Aggregated results from all validation checks."""

    model_config = ConfigDict(frozen=True)

    results: list[ValidationResult]

    @property
    def all_passed(self) -> bool:
        """Return ``True`` if every check passed."""
        return all(r.passed for r in self.results)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_game_counts(repo: Repository) -> list[ValidationResult]:
    """Validate that per-season game counts are within ±10% of the median.

    Season 2020 (COVID) is excluded from the median calculation but still
    validated — it is expected to flag as anomalous.
    """
    seasons = repo.get_seasons()
    if not seasons:
        return [
            ValidationResult(
                check_name="game_counts",
                passed=True,
                message="No seasons found — skipping game count check",
            )
        ]

    counts: dict[int, int] = {}
    for s in seasons:
        games = repo.get_games(s.year)
        counts[s.year] = len(games)

    # Compute median excluding COVID season.
    non_covid = [c for year, c in counts.items() if year != _COVID_SEASON]
    if not non_covid:
        return [
            ValidationResult(
                check_name="game_counts",
                passed=True,
                message="Only COVID season present — skipping game count check",
            )
        ]

    median_count = statistics.median(non_covid)
    lo = median_count * (1 - _GAME_COUNT_TOLERANCE)
    hi = median_count * (1 + _GAME_COUNT_TOLERANCE)

    anomalies: dict[str, int] = {}
    for year, count in sorted(counts.items()):
        if not (lo <= count <= hi):
            anomalies[str(year)] = count

    if anomalies:
        return [
            ValidationResult(
                check_name="game_counts",
                passed=False,
                message=(f"{len(anomalies)} season(s) outside ±10% of median ({median_count:.0f})"),
                details={"anomalies": anomalies, "median": median_count},
            )
        ]

    return [
        ValidationResult(
            check_name="game_counts",
            passed=True,
            message=(f"All {len(counts)} seasons within ±10% of median ({median_count:.0f})"),
        )
    ]


def _check_duplicate_games(repo: Repository) -> list[ValidationResult]:
    """Detect duplicate games within each season.

    Duplicate key: ``(season, day_num, w_team_id, l_team_id)``.
    """
    seasons = repo.get_seasons()
    if not seasons:
        return [
            ValidationResult(
                check_name="duplicate_games",
                passed=True,
                message="No seasons found — skipping duplicate check",
            )
        ]

    total_dupes = 0
    season_dupes: dict[str, int] = {}

    for s in seasons:
        games = repo.get_games(s.year)
        keys = [(g.season, g.day_num, g.w_team_id, g.l_team_id) for g in games]
        counter = Counter(keys)
        dupes = sum(c - 1 for c in counter.values() if c > 1)
        if dupes > 0:
            total_dupes += dupes
            season_dupes[str(s.year)] = dupes

    if total_dupes > 0:
        return [
            ValidationResult(
                check_name="duplicate_games",
                passed=False,
                message=f"{total_dupes} duplicate game(s) found",
                details={"duplicates_per_season": season_dupes},
            )
        ]

    return [
        ValidationResult(
            check_name="duplicate_games",
            passed=True,
            message="No duplicate games found",
        )
    ]


def _check_team_references(repo: Repository) -> list[ValidationResult]:
    """Check that all team IDs in games reference valid teams."""
    teams = repo.get_teams()
    team_ids = {t.team_id for t in teams}

    if not team_ids:
        return [
            ValidationResult(
                check_name="team_references",
                passed=True,
                message="No teams found — skipping team reference check",
            )
        ]

    seasons = repo.get_seasons()
    orphans: dict[str, list[int]] = {}

    for s in seasons:
        games = repo.get_games(s.year)
        season_orphans: set[int] = set()
        for g in games:
            if g.w_team_id not in team_ids:
                season_orphans.add(g.w_team_id)
            if g.l_team_id not in team_ids:
                season_orphans.add(g.l_team_id)
        if season_orphans:
            orphans[str(s.year)] = sorted(season_orphans)

    if orphans:
        unique_orphan_ids = {tid for ids in orphans.values() for tid in ids}
        return [
            ValidationResult(
                check_name="team_references",
                passed=False,
                message=(
                    f"{len(unique_orphan_ids)} unique orphan team ID(s) found across {len(orphans)} season(s)"
                ),
                details={"orphans_per_season": orphans},
            )
        ]

    return [
        ValidationResult(
            check_name="team_references",
            passed=True,
            message="All team references valid",
        )
    ]


# ---------------------------------------------------------------------------
# Top-level validation entry point
# ---------------------------------------------------------------------------


def validate_sync(repo: Repository) -> ValidationReport:
    """Run all post-sync validation checks and return a report.

    This function is **non-fatal** — it never raises on validation failures.
    Unexpected I/O errors (e.g., corrupt Parquet) may still propagate.
    The caller is responsible for logging results.
    """
    results: list[ValidationResult] = []
    results.extend(_check_game_counts(repo))
    results.extend(_check_duplicate_games(repo))
    results.extend(_check_team_references(repo))

    report = ValidationReport(results=results)

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    if report.all_passed:
        log.info("[validation] %d/%d checks passed", passed, total)
    else:
        log.info("[validation] %d/%d checks passed, %d warning(s)", passed, total, total - passed)
        for r in results:
            if not r.passed:
                log.warning("[validation] %s: %s", r.check_name, r.message)

    return report
