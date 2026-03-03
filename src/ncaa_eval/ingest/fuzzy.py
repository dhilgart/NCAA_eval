"""Centralized fuzzy team-name matching utility.

Provides a single public function for resolving team names against a
candidate mapping, used by both the ESPN connector and the sync engine.
"""

from __future__ import annotations

from rapidfuzz import fuzz


def fuzzy_match_team(
    name: str,
    candidates: dict[str, int],
    threshold: int = 80,
) -> int | None:
    """Match a team name to a candidate mapping using fuzzy lookup.

    Applies ``rapidfuzz.fuzz.token_set_ratio`` (case-insensitive) against all
    *candidates* and returns the best match whose score meets the *threshold*.
    Callers are responsible for attempting exact matches before calling this
    function.

    Args:
        name: Team name to match.
        candidates: Mapping of known names to team IDs.
        threshold: Minimum rapidfuzz token_set_ratio score (0-100).

    Returns:
        Matched team ID, or ``None`` if no match meets the threshold.
    """
    lower_name = name.lower()
    best_score = 0.0
    best_id: int | None = None
    for candidate_name, tid in candidates.items():
        score = fuzz.token_set_ratio(lower_name, candidate_name.lower())
        if score > best_score:
            best_score = score
            best_id = tid

    if best_score >= threshold and best_id is not None:
        return best_id

    return None
