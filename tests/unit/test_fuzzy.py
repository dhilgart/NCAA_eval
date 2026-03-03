"""Unit tests for ncaa_eval.ingest.fuzzy (fuzzy_match_team)."""

from __future__ import annotations

import pytest

from ncaa_eval.ingest.fuzzy import fuzzy_match_team

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CANDIDATES: dict[str, int] = {
    "Duke": 1181,
    "North Carolina": 1314,
    "Kentucky": 1243,
    "Kansas": 1242,
}


# ---------------------------------------------------------------------------
# TestFuzzyMatchTeam
# ---------------------------------------------------------------------------


class TestFuzzyMatchTeam:
    """Tests for the fuzzy_match_team utility."""

    @pytest.mark.smoke
    def test_exact_match(self) -> None:
        assert fuzzy_match_team("Duke", _CANDIDATES) == 1181

    def test_case_insensitive_exact(self) -> None:
        assert fuzzy_match_team("duke", _CANDIDATES) == 1181

    def test_case_insensitive_mixed(self) -> None:
        assert fuzzy_match_team("NORTH CAROLINA", _CANDIDATES) == 1314

    def test_fuzzy_match_above_threshold(self) -> None:
        # "Kentucky Wildcats" should fuzzy-match "Kentucky" at a high score.
        result = fuzzy_match_team("Kentucky Wildcats", _CANDIDATES)
        assert result == 1243

    def test_no_match_below_threshold(self) -> None:
        result = fuzzy_match_team("Completely Unknown University", _CANDIDATES)
        assert result is None

    def test_custom_threshold_strict(self) -> None:
        # With threshold=100, only perfect fuzzy matches pass.
        # "Totally Different Name" won't reach 100 against any candidate.
        result = fuzzy_match_team("Totally Different Name", _CANDIDATES, threshold=100)
        assert result is None

    def test_custom_threshold_lenient(self) -> None:
        # With very low threshold, even poor matches pass.
        result = fuzzy_match_team("K", _CANDIDATES, threshold=1)
        assert result is not None  # Some candidate will score > 1

    def test_empty_candidates(self) -> None:
        result = fuzzy_match_team("Duke", {})
        assert result is None

    def test_empty_name(self) -> None:
        result = fuzzy_match_team("", _CANDIDATES)
        # rapidfuzz.fuzz.token_set_ratio("", any_string) == 0.0, which is below
        # the default threshold of 80, so empty input always returns None.
        assert result is None
