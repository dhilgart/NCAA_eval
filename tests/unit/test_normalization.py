"""Unit tests for ncaa_eval.transform.normalization."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from ncaa_eval.transform.normalization import (
    _FALLBACK_SYSTEMS,
    _PRIMARY_COMPOSITE,
    ConferenceLookup,
    CoverageGateResult,
    MasseyOrdinalsStore,
    TeamNameNormalizer,
    TourneySeedTable,
    parse_seed,
)

# ---------------------------------------------------------------------------
# Fixtures — CSV helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def spellings_csv(tmp_path: Path) -> Path:
    """MTeamSpellings.csv fixture with a handful of known variants."""
    content = textwrap.dedent("""\
        TeamNameSpelling,TeamID
        Duke,1181
        Blue Devils,1181
        UNLV,1314
        Nevada Las Vegas,1314
        Kansas,1242
        KU,1242
        connecticut,1112
        UConn,1112
    """)
    p = tmp_path / "MTeamSpellings.csv"
    p.write_text(content)
    return p


@pytest.fixture
def seeds_csv(tmp_path: Path) -> Path:
    """MNCAATourneySeeds.csv fixture with regular and play-in seeds."""
    content = textwrap.dedent("""\
        Season,Seed,TeamID
        2023,W01,1181
        2023,X16a,1314
        2023,Y11b,1242
        2022,Z03,1112
    """)
    p = tmp_path / "MNCAATourneySeeds.csv"
    p.write_text(content)
    return p


@pytest.fixture
def conferences_csv(tmp_path: Path) -> Path:
    """MTeamConferences.csv fixture."""
    content = textwrap.dedent("""\
        Season,TeamID,ConfAbbrev
        2023,1181,ACC
        2023,1314,MWC
        2022,1112,BE
    """)
    p = tmp_path / "MTeamConferences.csv"
    p.write_text(content)
    return p


def _build_massey_csv(tmp_path: Path, rows: list[tuple[int, int, str, int, int]]) -> Path:
    """Build a MMasseyOrdinals.csv from a list of (Season, RankingDayNum, SystemName, TeamID, OrdinalRank) tuples."""
    header = "Season,RankingDayNum,SystemName,TeamID,OrdinalRank\n"
    body = "\n".join(f"{s},{d},{sys},{t},{r}" for s, d, sys, t, r in rows)
    p = tmp_path / "MMasseyOrdinals.csv"
    p.write_text(header + body + "\n")
    return p


def _all_seasons_rows(
    systems: list[str], team_id: int = 1001, rank: int = 5
) -> list[tuple[int, int, str, int, int]]:
    """Generate one row per season 2003–2025 for each system."""
    rows: list[tuple[int, int, str, int, int]] = []
    for season in range(2003, 2026):
        for sys in systems:
            rows.append((season, 100, sys, team_id, rank))
    return rows


# ---------------------------------------------------------------------------
# parse_seed tests (7.1–7.3)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.smoke
def test_parse_seed_regular() -> None:
    """7.1: 'W01' → region='W', seed_num=1, is_play_in=False."""
    ts = parse_seed(2023, 1181, "W01")
    assert ts.region == "W"
    assert ts.seed_num == 1
    assert ts.is_play_in is False
    assert ts.season == 2023
    assert ts.team_id == 1181
    assert ts.seed_str == "W01"


@pytest.mark.unit
@pytest.mark.smoke
def test_parse_seed_play_in_a() -> None:
    """7.2: 'X16a' → region='X', seed_num=16, is_play_in=True."""
    ts = parse_seed(2023, 1314, "X16a")
    assert ts.region == "X"
    assert ts.seed_num == 16
    assert ts.is_play_in is True


@pytest.mark.unit
@pytest.mark.smoke
def test_parse_seed_play_in_b() -> None:
    """7.3: 'Y11b' → is_play_in=True, seed_num=11."""
    ts = parse_seed(2023, 1242, "Y11b")
    assert ts.is_play_in is True
    assert ts.seed_num == 11
    assert ts.region == "Y"


@pytest.mark.unit
def test_parse_seed_invalid_short() -> None:
    """parse_seed raises ValueError for seed_str shorter than 3 chars."""
    with pytest.raises(ValueError, match="Invalid seed string"):
        parse_seed(2023, 1181, "W1")


# ---------------------------------------------------------------------------
# TourneySeedTable tests (7.4–7.5)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.smoke
def test_tourney_seed_table_from_csv(seeds_csv: Path) -> None:
    """7.4: TourneySeedTable.from_csv loads fixture CSV, returns correct TourneySeed."""
    table = TourneySeedTable.from_csv(seeds_csv)
    ts = table.get(2023, 1181)
    assert ts is not None
    assert ts.region == "W"
    assert ts.seed_num == 1
    assert ts.is_play_in is False
    # Play-in seeds also loaded correctly
    ts_playin = table.get(2023, 1314)
    assert ts_playin is not None
    assert ts_playin.is_play_in is True


@pytest.mark.unit
@pytest.mark.smoke
def test_tourney_seed_table_get_missing(seeds_csv: Path) -> None:
    """7.5: TourneySeedTable.get returns None for unknown (season, team_id)."""
    table = TourneySeedTable.from_csv(seeds_csv)
    assert table.get(2023, 9999) is None
    assert table.get(2010, 1181) is None


@pytest.mark.unit
def test_tourney_seed_table_all_seeds(seeds_csv: Path) -> None:
    """all_seeds with season filter returns only that season's seeds."""
    table = TourneySeedTable.from_csv(seeds_csv)
    seeds_2023 = table.all_seeds(season=2023)
    assert len(seeds_2023) == 3
    assert all(ts.season == 2023 for ts in seeds_2023)
    seeds_all = table.all_seeds()
    assert len(seeds_all) == 4


# ---------------------------------------------------------------------------
# TeamNameNormalizer tests (7.6–7.9)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.smoke
def test_team_name_normalizer_exact_match(spellings_csv: Path) -> None:
    """7.6: normalize exact match (case-insensitive) returns correct TeamID."""
    normalizer = TeamNameNormalizer.from_csv(spellings_csv)
    assert normalizer.normalize("Duke") == 1181
    assert normalizer.normalize("duke") == 1181
    assert normalizer.normalize("DUKE") == 1181
    assert normalizer.normalize("Kansas") == 1242


@pytest.mark.unit
@pytest.mark.smoke
def test_team_name_normalizer_known_variant(spellings_csv: Path) -> None:
    """7.7: UNLV variant resolves correctly to Nevada Las Vegas TeamID."""
    normalizer = TeamNameNormalizer.from_csv(spellings_csv)
    assert normalizer.normalize("UNLV") == 1314
    assert normalizer.normalize("Nevada Las Vegas") == 1314


@pytest.mark.unit
@pytest.mark.smoke
def test_team_name_normalizer_unknown_returns_none(spellings_csv: Path) -> None:
    """7.8: Unknown name returns None and logs a warning (no exception raised)."""
    from unittest.mock import patch

    normalizer = TeamNameNormalizer.from_csv(spellings_csv)
    with patch("ncaa_eval.transform.normalization.logger") as mock_logger:
        result = normalizer.normalize("XYZ Unknown Team")
    assert result is None
    mock_logger.warning.assert_called_once()
    warning_fmt = mock_logger.warning.call_args[0][0]
    assert "no match for" in warning_fmt


@pytest.mark.unit
@pytest.mark.smoke
def test_team_name_normalizer_idempotent(spellings_csv: Path) -> None:
    """7.9: Calling normalize(name) twice returns same value."""
    normalizer = TeamNameNormalizer.from_csv(spellings_csv)
    first = normalizer.normalize("Duke")
    second = normalizer.normalize("Duke")
    assert first == second == 1181
    first_miss = normalizer.normalize("NoSuchTeam")
    second_miss = normalizer.normalize("NoSuchTeam")
    assert first_miss == second_miss is None


# ---------------------------------------------------------------------------
# ConferenceLookup tests (7.10–7.11)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.smoke
def test_conference_lookup_get_known(conferences_csv: Path) -> None:
    """7.10: ConferenceLookup.get returns correct conference for known (season, team_id)."""
    lookup = ConferenceLookup.from_csv(conferences_csv)
    assert lookup.get(2023, 1181) == "ACC"
    assert lookup.get(2023, 1314) == "MWC"
    assert lookup.get(2022, 1112) == "BE"


@pytest.mark.unit
@pytest.mark.smoke
def test_conference_lookup_get_missing(conferences_csv: Path) -> None:
    """7.11: ConferenceLookup.get returns None for missing (season, team_id)."""
    lookup = ConferenceLookup.from_csv(conferences_csv)
    assert lookup.get(2023, 9999) is None
    assert lookup.get(2010, 1181) is None


# ---------------------------------------------------------------------------
# MasseyOrdinalsStore — coverage gate tests (7.12–7.13)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_massey_coverage_gate_no_fallback(tmp_path: Path) -> None:
    """7.12: Gate with SAG+WLK for all 2003–2025 → fallback_used=False, SAG/WLK in recommended."""
    rows = _all_seasons_rows(["SAG", "WLK", "POM", "MOR"])
    csv = _build_massey_csv(tmp_path, rows)
    store = MasseyOrdinalsStore.from_csv(csv)
    result = store.run_coverage_gate()
    assert result.fallback_used is False
    assert "SAG" in result.recommended_systems
    assert "WLK" in result.recommended_systems
    assert result.fallback_reason == ""


@pytest.mark.unit
def test_massey_coverage_gate_fallback(tmp_path: Path) -> None:
    """7.13: Gate with WLK missing for some seasons → fallback_used=True, recommended=MOR+POM+DOL."""
    # Only SAG for all seasons; WLK only for a subset
    rows: list[tuple[int, int, str, int, int]] = []
    for season in range(2003, 2026):
        rows.append((season, 100, "SAG", 1001, 5))
    # WLK only for 2010–2020 (missing 2003–2009 and 2021–2025)
    for season in range(2010, 2021):
        rows.append((season, 100, "WLK", 1001, 10))
    csv = _build_massey_csv(tmp_path, rows)
    store = MasseyOrdinalsStore.from_csv(csv)
    result = store.run_coverage_gate()
    assert result.fallback_used is True
    assert set(result.recommended_systems) == set(_FALLBACK_SYSTEMS)
    assert "WLK" in result.fallback_reason


# ---------------------------------------------------------------------------
# MasseyOrdinalsStore — get_snapshot temporal filtering (7.14–7.15)
# ---------------------------------------------------------------------------


@pytest.fixture
def massey_temporal_store(tmp_path: Path) -> MasseyOrdinalsStore:
    """Store with SAG rankings at day 80, 100, 130 for team 1001 in season 2023."""
    rows: list[tuple[int, int, str, int, int]] = [
        (2023, 80, "SAG", 1001, 20),
        (2023, 100, "SAG", 1001, 15),
        (2023, 130, "SAG", 1001, 10),
        (2023, 100, "SAG", 1002, 5),
        (2023, 130, "SAG", 1002, 3),
    ]
    csv = _build_massey_csv(tmp_path, rows)
    return MasseyOrdinalsStore.from_csv(csv)


@pytest.mark.unit
def test_massey_get_snapshot_temporal_filter(massey_temporal_store: MasseyOrdinalsStore) -> None:
    """7.14: get_snapshot only returns ordinals with RankingDayNum ≤ day_num."""
    snap = massey_temporal_store.get_snapshot(2023, day_num=100, systems=["SAG"])
    # day 130 must NOT appear
    assert not snap.empty
    # team 1001 should have rank 15 (day 100), not 10 (day 130)
    assert snap.loc[1001, "SAG"] == 15


@pytest.mark.unit
def test_massey_get_snapshot_last_available(massey_temporal_store: MasseyOrdinalsStore) -> None:
    """7.15: get_snapshot uses last available RankingDayNum per (system, team) before cutoff."""
    snap = massey_temporal_store.get_snapshot(2023, day_num=90, systems=["SAG"])
    # Only day 80 is ≤ 90; rank should be 20 (not 15 from day 100)
    assert snap.loc[1001, "SAG"] == 20
    # team 1002 has no record ≤ 90
    assert 1002 not in snap.index


# ---------------------------------------------------------------------------
# MasseyOrdinalsStore — composite methods (7.16–7.17)
# ---------------------------------------------------------------------------


@pytest.fixture
def massey_composite_store(tmp_path: Path) -> MasseyOrdinalsStore:
    """Fixture with 2 systems (SAG, POM), 3 teams for season 2023 at day 100."""
    rows: list[tuple[int, int, str, int, int]] = [
        (2023, 100, "SAG", 1001, 10),
        (2023, 100, "SAG", 1002, 20),
        (2023, 100, "SAG", 1003, 30),
        (2023, 100, "POM", 1001, 5),
        (2023, 100, "POM", 1002, 15),
        (2023, 100, "POM", 1003, 25),
    ]
    csv = _build_massey_csv(tmp_path, rows)
    return MasseyOrdinalsStore.from_csv(csv)


@pytest.mark.unit
def test_massey_composite_simple_average(massey_composite_store: MasseyOrdinalsStore) -> None:
    """7.16: composite_simple_average: 2 systems, 3 teams → correct mean per team."""
    result = massey_composite_store.composite_simple_average(2023, 100, ["SAG", "POM"])
    assert result[1001] == pytest.approx(7.5)  # (10 + 5) / 2
    assert result[1002] == pytest.approx(17.5)  # (20 + 15) / 2
    assert result[1003] == pytest.approx(27.5)  # (30 + 25) / 2


@pytest.mark.unit
def test_massey_composite_weighted(massey_composite_store: MasseyOrdinalsStore) -> None:
    """7.17: composite_weighted: correct weighted average with known weights."""
    # weights: SAG=0.75, POM=0.25 (normalized they are 0.75 and 0.25 already)
    result = massey_composite_store.composite_weighted(2023, 100, weights={"SAG": 0.75, "POM": 0.25})
    # team 1001: 0.75 * 10 + 0.25 * 5 = 7.5 + 1.25 = 8.75
    assert result[1001] == pytest.approx(8.75)
    # team 1002: 0.75 * 20 + 0.25 * 15 = 15 + 3.75 = 18.75
    assert result[1002] == pytest.approx(18.75)


# ---------------------------------------------------------------------------
# MasseyOrdinalsStore — pre_tournament_snapshot (7.18)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_massey_pre_tournament_snapshot(tmp_path: Path) -> None:
    """7.18: pre_tournament_snapshot uses only RankingDayNum ≤ 128; day 135 excluded."""
    rows: list[tuple[int, int, str, int, int]] = [
        (2023, 100, "SAG", 1001, 20),
        (2023, 128, "SAG", 1001, 15),
        (2023, 135, "SAG", 1001, 10),  # After Selection Sunday — must be excluded
    ]
    csv = _build_massey_csv(tmp_path, rows)
    store = MasseyOrdinalsStore.from_csv(csv)
    snap = store.pre_tournament_snapshot(2023, systems=["SAG"])
    # Should use rank 15 (day 128), NOT 10 (day 135)
    assert snap.loc[1001, "SAG"] == 15


# ---------------------------------------------------------------------------
# MasseyOrdinalsStore — normalization methods (7.19–7.20)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_massey_normalize_percentile(tmp_path: Path) -> None:
    """7.19: normalize_percentile returns values in [0, 1] range.

    Uses realistic ranks (1, 2, 3) for 3 teams so that rank/n_teams is bounded [0, 1].
    """
    # Ranks run 1-to-n so that rank/n_teams is naturally [0, 1]
    rows: list[tuple[int, int, str, int, int]] = [
        (2023, 100, "SAG", 1001, 1),
        (2023, 100, "SAG", 1002, 2),
        (2023, 100, "SAG", 1003, 3),
    ]
    csv = _build_massey_csv(tmp_path, rows)
    store = MasseyOrdinalsStore.from_csv(csv)
    result = store.normalize_percentile(2023, 100, "SAG")
    assert all(0.0 <= v <= 1.0 for v in result.values)


@pytest.mark.unit
def test_massey_normalize_zscore(massey_composite_store: MasseyOrdinalsStore) -> None:
    """7.20: normalize_zscore: mean ≈ 0, std ≈ 1 over fixture data."""
    result = massey_composite_store.normalize_zscore(2023, 100, "SAG")
    assert result.mean() == pytest.approx(0.0, abs=1e-10)
    assert result.std() == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# MasseyOrdinalsStore — composite_pca (AC 8c)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_massey_composite_pca_explicit_components(massey_composite_store: MasseyOrdinalsStore) -> None:
    """composite_pca with explicit n_components returns DataFrame with PC columns indexed by TeamID."""
    result = massey_composite_store.composite_pca(2023, 100, n_components=1)
    assert "PC1" in result.columns
    assert len(result) == 3
    assert set(result.index) == {1001, 1002, 1003}


@pytest.mark.unit
def test_massey_composite_pca_auto_select(massey_composite_store: MasseyOrdinalsStore) -> None:
    """composite_pca with n_components=None auto-selects to capture >=90% variance."""
    result = massey_composite_store.composite_pca(2023, 100, n_components=None, min_variance=0.90)
    assert result.shape[0] == 3
    assert all(col.startswith("PC") for col in result.columns)


@pytest.mark.unit
def test_massey_composite_pca_empty_snapshot(tmp_path: Path) -> None:
    """composite_pca returns empty DataFrame when no data matches season/day_num."""
    rows: list[tuple[int, int, str, int, int]] = [
        (2023, 100, "SAG", 1001, 10),
    ]
    csv = _build_massey_csv(tmp_path, rows)
    store = MasseyOrdinalsStore.from_csv(csv)
    result = store.composite_pca(2024, 100)  # Wrong season → empty snapshot
    assert result.empty


@pytest.mark.unit
def test_massey_composite_weighted_empty_weights_raises(massey_composite_store: MasseyOrdinalsStore) -> None:
    """composite_weighted raises ValueError when weights dict is empty."""
    with pytest.raises(ValueError, match="weights dict must not be empty"):
        massey_composite_store.composite_weighted(2023, 100, weights={})


# ---------------------------------------------------------------------------
# MasseyOrdinalsStore — rank delta
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.smoke
def test_massey_normalize_rank_delta(massey_composite_store: MasseyOrdinalsStore) -> None:
    """normalize_rank_delta returns correct rank delta for matchup."""
    snap = massey_composite_store.get_snapshot(2023, 100, systems=["SAG"])
    # team 1001 rank=10, team 1002 rank=20 → delta = 10-20 = -10
    delta = massey_composite_store.normalize_rank_delta(snap, 1001, 1002, "SAG")
    assert delta == pytest.approx(-10.0)
    # Reversed: delta = 20-10 = +10 (team_a ranked worse)
    delta_rev = massey_composite_store.normalize_rank_delta(snap, 1002, 1001, "SAG")
    assert delta_rev == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# MasseyOrdinalsStore — empty snapshot
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_massey_get_snapshot_empty(tmp_path: Path) -> None:
    """get_snapshot returns empty DataFrame when no records match filters."""
    rows: list[tuple[int, int, str, int, int]] = [
        (2023, 100, "SAG", 1001, 10),
    ]
    csv = _build_massey_csv(tmp_path, rows)
    store = MasseyOrdinalsStore.from_csv(csv)
    # Wrong season
    snap = store.get_snapshot(2024, 100)
    assert snap.empty
    # day_num too early
    snap2 = store.get_snapshot(2023, 50)
    assert snap2.empty


# ---------------------------------------------------------------------------
# CoverageGateResult — dataclass immutability
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.smoke
def test_coverage_gate_result_frozen() -> None:
    """CoverageGateResult is a frozen dataclass (immutable)."""
    result = CoverageGateResult(
        primary_systems=_PRIMARY_COMPOSITE,
        fallback_used=False,
        fallback_reason="",
        recommended_systems=_PRIMARY_COMPOSITE,
    )
    with pytest.raises((AttributeError, TypeError)):
        result.fallback_used = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TourneySeed — dataclass immutability
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.smoke
def test_tourney_seed_frozen() -> None:
    """TourneySeed is a frozen dataclass (immutable)."""
    ts = parse_seed(2023, 1181, "W01")
    with pytest.raises((AttributeError, TypeError)):
        ts.seed_num = 5  # type: ignore[misc]
