"""Canonical team ID mapping, lookup tables, and Massey Ordinals ingestion.

Provides normalization and lookup infrastructure for the feature pipeline:

* :class:`TeamNameNormalizer` — maps diverse team name spellings to canonical
  ``TeamID`` integers using ``MTeamSpellings.csv``.
* :class:`TourneySeedTable` — wraps ``MNCAATourneySeeds.csv`` into a structured
  ``(season, team_id) → TourneySeed`` lookup.
* :class:`ConferenceLookup` — wraps ``MTeamConferences.csv`` into a
  ``(season, team_id) → conf_abbrev`` lookup.
* :class:`MasseyOrdinalsStore` — DataFrame-backed store for ``MMasseyOrdinals.csv``
  with temporal filtering, coverage gate, and composite computation methods.

Design invariants:
- No imports from ``ncaa_eval.ingest`` — this module is a pure CSV-loading layer.
- No ``df.iterrows()`` — vectorized pandas operations throughout; ``itertuples``
  is acceptable only for non-vectorizable dict construction with string parsing.
- ``mypy --strict`` compliant: all types fully annotated, no bare ``Any``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants for MasseyOrdinalsStore coverage gate
# ---------------------------------------------------------------------------

_MASSEY_FIRST_SEASON: int = 2003
_MASSEY_LAST_SEASON: int = 2025
_MASSEY_ALL_SEASONS: frozenset[int] = frozenset(range(_MASSEY_FIRST_SEASON, _MASSEY_LAST_SEASON + 1))
_GATE_SYSTEMS: tuple[str, ...] = ("SAG", "WLK")
_FALLBACK_SYSTEMS: tuple[str, ...] = ("MOR", "POM", "DOL")
_PRIMARY_COMPOSITE: tuple[str, ...] = ("SAG", "POM", "MOR", "WLK")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TourneySeed:
    """Structured representation of a single NCAA Tournament seed entry.

    Attributes:
        season: Season year (e.g., 2023).
        team_id: Canonical Kaggle TeamID integer.
        seed_str: Raw seed string as it appears in ``MNCAATourneySeeds.csv``
            (e.g., ``"W01"``, ``"X11a"``).
        region: Single-character region code: W, X, Y, or Z.
        seed_num: Seed number 1–16.
        is_play_in: True when the seed has an ``'a'`` or ``'b'`` suffix,
            indicating a First Four play-in game.
    """

    season: int
    team_id: int
    seed_str: str
    region: str
    seed_num: int
    is_play_in: bool


@dataclass(frozen=True)
class CoverageGateResult:
    """Result of the Massey Ordinals coverage gate check.

    Attributes:
        primary_systems: The four primary composite systems
            (SAG, POM, MOR, WLK).
        fallback_used: True when SAG or WLK are missing for one or more
            seasons 2003–2025 and the fallback composite is recommended.
        fallback_reason: Human-readable description of why the fallback was
            triggered (empty string when ``fallback_used=False``).
        recommended_systems: The system names the caller should use for
            composite computation — either the primary composite or the
            confirmed-full-coverage fallback (MOR, POM, DOL).
    """

    primary_systems: tuple[str, ...]
    fallback_used: bool
    fallback_reason: str
    recommended_systems: tuple[str, ...]


# ---------------------------------------------------------------------------
# Module-level function
# ---------------------------------------------------------------------------


def parse_seed(season: int, team_id: int, seed_str: str) -> TourneySeed:
    """Parse a raw tournament seed string into a structured :class:`TourneySeed`.

    Seed strings from ``MNCAATourneySeeds.csv`` follow the pattern
    ``[WXYZ][0-9]{2}[ab]?``:

    * ``"W01"``  → region="W", seed_num=1,  is_play_in=False
    * ``"X16a"`` → region="X", seed_num=16, is_play_in=True
    * ``"Y11b"`` → region="Y", seed_num=11, is_play_in=True

    Args:
        season: Season year.
        team_id: Canonical Kaggle TeamID.
        seed_str: Raw seed string (e.g., ``"W01"``, ``"X11a"``).

    Returns:
        Fully parsed :class:`TourneySeed`.

    Raises:
        ValueError: If ``seed_str`` is shorter than 3 characters.
    """
    if len(seed_str) < 3:
        msg = f"Invalid seed string: {seed_str!r}"
        raise ValueError(msg)
    region = seed_str[0]
    seed_num = int(seed_str[1:3])
    is_play_in = len(seed_str) > 3 and seed_str[3] in ("a", "b")
    return TourneySeed(
        season=season,
        team_id=team_id,
        seed_str=seed_str,
        region=region,
        seed_num=seed_num,
        is_play_in=is_play_in,
    )


# ---------------------------------------------------------------------------
# TeamNameNormalizer
# ---------------------------------------------------------------------------


class TeamNameNormalizer:
    """Maps diverse team name spellings to canonical ``TeamID`` integers.

    Wraps the ``MTeamSpellings.csv`` lookup table. Matching is
    case-insensitive. On a miss, a WARNING is logged with any close prefix
    matches and ``None`` is returned (no exception raised). The lookup is
    idempotent: calling :meth:`normalize` twice with the same input returns
    the same result.

    Args:
        spellings: Pre-lowercased mapping of ``team_name_spelling → team_id``.
    """

    def __init__(self, spellings: dict[str, int]) -> None:
        self._spellings = spellings

    @classmethod
    def from_csv(cls, path: Path) -> TeamNameNormalizer:
        """Construct from ``MTeamSpellings.csv``.

        Columns required: ``TeamNameSpelling``, ``TeamID``.

        Args:
            path: Path to ``MTeamSpellings.csv``.

        Returns:
            Initialised :class:`TeamNameNormalizer`.
        """
        df = pd.read_csv(path)
        spellings: dict[str, int] = dict(zip(df["TeamNameSpelling"].str.lower(), df["TeamID"].astype(int)))
        return cls(spellings)

    def normalize(self, name: str) -> int | None:
        """Look up *name* and return its canonical ``TeamID``, or ``None`` on miss.

        Args:
            name: Team name string (any case).

        Returns:
            Canonical ``TeamID`` integer, or ``None`` if not found.
        """
        key = name.lower()
        result = self._spellings.get(key)
        if result is not None:
            return result
        # Collect up to 5 prefix matches for the warning message
        prefix = key[:4]
        closest = {k: v for k, v in self._spellings.items() if k.startswith(prefix)}
        logger.warning(
            "TeamNameNormalizer: no match for %r; closest spellings: %s",
            name,
            list(closest.keys())[:5],
        )
        return None


# ---------------------------------------------------------------------------
# TourneySeedTable
# ---------------------------------------------------------------------------


class TourneySeedTable:
    """Lookup table for NCAA Tournament seeds by ``(season, team_id)``.

    Wraps ``MNCAATourneySeeds.csv`` into a dict-backed structure. Each seed
    is stored as a :class:`TourneySeed` frozen dataclass.

    Args:
        seeds: Mapping of ``(season, team_id) → TourneySeed``.
    """

    def __init__(self, seeds: dict[tuple[int, int], TourneySeed]) -> None:
        self._seeds = seeds

    @classmethod
    def from_csv(cls, path: Path) -> TourneySeedTable:
        """Construct from ``MNCAATourneySeeds.csv``.

        Columns required: ``Season``, ``Seed``, ``TeamID``.

        Uses ``itertuples`` (not ``iterrows``) for per-row string parsing —
        acceptable because the per-row operation (``parse_seed``) contains
        branching logic that cannot be vectorized.

        Args:
            path: Path to ``MNCAATourneySeeds.csv``.

        Returns:
            Initialised :class:`TourneySeedTable`.
        """
        df = pd.read_csv(path)
        seeds: dict[tuple[int, int], TourneySeed] = {}
        for season, seed_str, team_id in df[["Season", "Seed", "TeamID"]].itertuples(index=False):
            ts = parse_seed(int(season), int(team_id), str(seed_str))
            seeds[(ts.season, ts.team_id)] = ts
        return cls(seeds)

    def get(self, season: int, team_id: int) -> TourneySeed | None:
        """Return the :class:`TourneySeed` for ``(season, team_id)``, or ``None``.

        Args:
            season: Season year.
            team_id: Canonical Kaggle TeamID.

        Returns:
            Matching :class:`TourneySeed`, or ``None`` if not found.
        """
        return self._seeds.get((season, team_id))

    def all_seeds(self, season: int | None = None) -> list[TourneySeed]:
        """Return all stored seeds, optionally filtered to a single season.

        Args:
            season: If provided, only seeds for this season are returned.

        Returns:
            List of :class:`TourneySeed` objects.
        """
        if season is None:
            return list(self._seeds.values())
        return [ts for ts in self._seeds.values() if ts.season == season]


# ---------------------------------------------------------------------------
# ConferenceLookup
# ---------------------------------------------------------------------------


class ConferenceLookup:
    """Lookup table for team conference membership by ``(season, team_id)``.

    Wraps ``MTeamConferences.csv`` into a dict-backed structure.

    Args:
        lookup: Mapping of ``(season, team_id) → conf_abbrev``.
    """

    def __init__(self, lookup: dict[tuple[int, int], str]) -> None:
        self._lookup = lookup

    @classmethod
    def from_csv(cls, path: Path) -> ConferenceLookup:
        """Construct from ``MTeamConferences.csv``.

        Columns required: ``Season``, ``TeamID``, ``ConfAbbrev``.

        Args:
            path: Path to ``MTeamConferences.csv``.

        Returns:
            Initialised :class:`ConferenceLookup`.
        """
        df = pd.read_csv(path)
        lookup: dict[tuple[int, int], str] = {
            (int(s), int(t)): str(c) for s, t, c in zip(df["Season"], df["TeamID"], df["ConfAbbrev"])
        }
        return cls(lookup)

    def get(self, season: int, team_id: int) -> str | None:
        """Return the conference abbreviation for ``(season, team_id)``, or ``None``.

        Args:
            season: Season year.
            team_id: Canonical Kaggle TeamID.

        Returns:
            Conference abbreviation string, or ``None`` if not found.
        """
        return self._lookup.get((season, team_id))


# ---------------------------------------------------------------------------
# MasseyOrdinalsStore
# ---------------------------------------------------------------------------


class MasseyOrdinalsStore:
    """DataFrame-backed store for Massey Ordinal ranking systems.

    Ingests ``MMasseyOrdinals.csv`` and provides temporal filtering,
    coverage gate validation, composite computation (Options A–D), and
    per-system normalization.

    Args:
        df: Raw DataFrame with columns
            ``[Season, RankingDayNum, SystemName, TeamID, OrdinalRank]``.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    @classmethod
    def from_csv(cls, path: Path) -> MasseyOrdinalsStore:
        """Construct from ``MMasseyOrdinals.csv``.

        Columns required: ``Season``, ``RankingDayNum``, ``SystemName``,
        ``TeamID``, ``OrdinalRank``.

        Args:
            path: Path to ``MMasseyOrdinals.csv``.

        Returns:
            Initialised :class:`MasseyOrdinalsStore`.
        """
        df = pd.read_csv(
            path,
            dtype={
                "Season": int,
                "RankingDayNum": int,
                "SystemName": str,
                "TeamID": int,
                "OrdinalRank": int,
            },
        )
        return cls(df)

    def run_coverage_gate(self) -> CoverageGateResult:
        """Check whether SAG and WLK cover all seasons 2003–2025.

        If either system has gaps the fallback composite (MOR, POM, DOL) is
        recommended instead of the primary composite (SAG, POM, MOR, WLK).

        Returns:
            :class:`CoverageGateResult` with coverage findings and the
            recommended system list.
        """
        covered: dict[str, set[int]] = (
            self._df[self._df["SystemName"].isin(list(_GATE_SYSTEMS))]
            .groupby("SystemName")["Season"]
            .apply(set)
            .to_dict()
        )
        missing: list[str] = []
        for system in _GATE_SYSTEMS:
            system_seasons: set[int] = covered.get(system, set())
            if not _MASSEY_ALL_SEASONS.issubset(system_seasons):
                missing.append(system)
        if missing:
            return CoverageGateResult(
                primary_systems=_PRIMARY_COMPOSITE,
                fallback_used=True,
                fallback_reason=f"{missing} missing for some seasons 2003–2025",
                recommended_systems=_FALLBACK_SYSTEMS,
            )
        return CoverageGateResult(
            primary_systems=_PRIMARY_COMPOSITE,
            fallback_used=False,
            fallback_reason="",
            recommended_systems=_PRIMARY_COMPOSITE,
        )

    def get_snapshot(
        self,
        season: int,
        day_num: int,
        systems: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return wide-format ordinal ranks as of *day_num* for *season*.

        For each ``(SystemName, TeamID)`` pair, uses the latest
        ``RankingDayNum`` that is ``≤ day_num``.  Returns a DataFrame with
        ``TeamID`` as index and one column per ranking system.

        Args:
            season: Season year.
            day_num: Inclusive upper bound on ``RankingDayNum``.
            systems: If provided, only include these system names. ``None``
                returns all available systems.

        Returns:
            Wide-format DataFrame (index=TeamID, columns=SystemName). Empty
            DataFrame if no records satisfy the filters.
        """
        mask = (self._df["Season"] == season) & (self._df["RankingDayNum"] <= day_num)
        filtered = self._df[mask]
        if systems is not None:
            filtered = filtered[filtered["SystemName"].isin(systems)]
        if filtered.empty:
            return pd.DataFrame()
        # Keep last available RankingDayNum per (SystemName, TeamID)
        latest = filtered.loc[filtered.groupby(["SystemName", "TeamID"])["RankingDayNum"].idxmax()]
        pivot: pd.DataFrame = latest.pivot(index="TeamID", columns="SystemName", values="OrdinalRank")
        pivot.columns.name = None  # Remove MultiIndex name artifact
        return pivot

    def composite_simple_average(self, season: int, day_num: int, systems: list[str]) -> pd.Series:
        """Option A: simple average of ordinal ranks across *systems* per team.

        Args:
            season: Season year.
            day_num: Temporal cutoff (inclusive).
            systems: List of system names to average.

        Returns:
            Series indexed by TeamID with mean ordinal rank per team.
        """
        snapshot = self.get_snapshot(season, day_num, systems=systems)
        result: pd.Series = snapshot[systems].mean(axis=1)
        return result

    def composite_weighted(self, season: int, day_num: int, weights: dict[str, float]) -> pd.Series:
        """Option B: weighted average of ordinal ranks using caller-supplied weights.

        Weights are normalized to sum to 1 before computation.

        Args:
            season: Season year.
            day_num: Temporal cutoff (inclusive).
            weights: Mapping of system name → weight (any positive floats).

        Returns:
            Series indexed by TeamID with weighted ordinal rank per team.
        """
        systems = list(weights.keys())
        snapshot = self.get_snapshot(season, day_num, systems=systems)
        w = pd.Series(weights)
        w = w / w.sum()  # Normalize to sum=1
        result: pd.Series = snapshot[systems].mul(w.values).sum(axis=1)
        return result

    def composite_pca(
        self,
        season: int,
        day_num: int,
        n_components: int | None = None,
        min_variance: float = 0.90,
    ) -> pd.DataFrame:
        """Option C: PCA reduction of all available systems.

        When ``n_components=None``, automatically selects the minimum number
        of components needed to capture ``min_variance`` of total variance.

        Args:
            season: Season year.
            day_num: Temporal cutoff (inclusive).
            n_components: Number of principal components to retain. ``None``
                triggers automatic selection based on ``min_variance``.
            min_variance: Minimum cumulative explained variance required when
                ``n_components=None`` (default 0.90 = 90%).

        Returns:
            DataFrame with columns ``PC1, PC2, ...`` indexed by TeamID.
            Rows with any NaN system value are dropped before PCA.
        """
        from sklearn.decomposition import PCA  # type: ignore[import-untyped]

        snapshot = self.get_snapshot(season, day_num)
        snapshot = snapshot.dropna()
        if n_components is None:
            pca_full = PCA()
            pca_full.fit(snapshot.values)
            cumvar = pca_full.explained_variance_ratio_.cumsum()
            n_components = int((cumvar >= min_variance).argmax()) + 1
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(snapshot.values)
        cols = [f"PC{i + 1}" for i in range(n_components)]
        return pd.DataFrame(components, index=snapshot.index, columns=cols)

    def pre_tournament_snapshot(self, season: int, systems: list[str] | None = None) -> pd.DataFrame:
        """Option D: pre-tournament snapshot using ordinals from ``RankingDayNum ≤ 128``.

        DayNum 128 corresponds approximately to Selection Sunday in the Kaggle
        calendar. Only ordinals available before the tournament begins are used.

        Args:
            season: Season year.
            systems: If provided, only include these system names.

        Returns:
            Wide-format DataFrame in the same structure as :meth:`get_snapshot`.
        """
        return self.get_snapshot(season, day_num=128, systems=systems)

    def normalize_rank_delta(self, snapshot: pd.DataFrame, team_a: int, team_b: int, system: str) -> float:
        """Return ordinal rank delta for a matchup between *team_a* and *team_b*.

        A positive result means *team_a* is ranked worse (higher rank number =
        worse) than *team_b* in this system.

        Args:
            snapshot: Wide-format snapshot DataFrame (index=TeamID,
                columns=SystemName) from :meth:`get_snapshot`.
            team_a: First team's canonical TeamID.
            team_b: Second team's canonical TeamID.
            system: System name column to use.

        Returns:
            ``snapshot.loc[team_a, system] - snapshot.loc[team_b, system]``
        """
        return float(snapshot.loc[team_a, system] - snapshot.loc[team_b, system])

    def normalize_percentile(self, season: int, day_num: int, system: str) -> pd.Series:
        """Return per-season percentile rank for *system* bounded to ``[0, 1]``.

        Computed as ``OrdinalRank / n_teams`` where ``n_teams`` is the number
        of teams with a rank in this season/system snapshot.

        Args:
            season: Season year.
            day_num: Temporal cutoff (inclusive).
            system: System name.

        Returns:
            Series indexed by TeamID with percentile values in ``[0, 1]``.
        """
        snap = self.get_snapshot(season, day_num, systems=[system])
        col: pd.Series = snap[system]
        n_teams = col.count()
        result: pd.Series = col / n_teams
        return result

    def normalize_zscore(self, season: int, day_num: int, system: str) -> pd.Series:
        """Return per-season z-score for *system*.

        Computed as ``(rank - mean_rank) / std_rank`` across all teams in the
        snapshot.

        Args:
            season: Season year.
            day_num: Temporal cutoff (inclusive).
            system: System name.

        Returns:
            Series indexed by TeamID with z-score values (mean ≈ 0, std ≈ 1).
        """
        snap = self.get_snapshot(season, day_num, systems=[system])
        col: pd.Series = snap[system]
        result: pd.Series = (col - col.mean()) / col.std()
        return result
