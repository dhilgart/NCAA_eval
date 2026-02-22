"""Game-by-game Elo rating engine for NCAA basketball feature engineering.

Computes Elo ratings as a **feature building block** — the resulting per-team
ratings feed into models (XGBoost, etc.) as input features.  This module does
NOT implement model-level ``train``/``predict``/``save`` interfaces; those
belong in Story 5.3.

Key design points:

* ``update_game()`` returns the **before** ratings, then mutates internal
  state, guaranteeing walk-forward temporal safety.
* Variable K-factor: early-season → regular-season → tournament.
* Margin-of-victory scaling with diminishing returns (Silver/SBCB formula).
* Home-court adjustment subtracted from effective rating before computing
  expected outcome.
* Season mean-reversion toward conference mean (or global mean as fallback).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd  # type: ignore[import-untyped]

from ncaa_eval.transform.serving import rescale_overtime

if TYPE_CHECKING:
    from ncaa_eval.ingest.schema import Game
    from ncaa_eval.transform.normalization import ConferenceLookup

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EloConfig:
    """Frozen configuration for the Elo feature engine.

    All K-factor, margin scaling, home-court, and mean-reversion parameters
    are configurable with sensible defaults matching the Silver/SBCB model.
    """

    initial_rating: float = 1500.0
    k_early: float = 56.0
    k_regular: float = 38.0
    k_tournament: float = 47.5
    early_game_threshold: int = 20
    margin_exponent: float = 0.85
    max_margin: int = 25
    home_advantage_elo: float = 3.5
    mean_reversion_fraction: float = 0.25


# ── Engine ───────────────────────────────────────────────────────────────────


class EloFeatureEngine:
    """Game-by-game Elo rating engine.

    Parameters
    ----------
    config
        Frozen Elo configuration.
    conference_lookup
        Optional conference lookup for season mean-reversion.  When ``None``,
        mean-reversion falls back to global mean.
    """

    def __init__(
        self,
        config: EloConfig,
        conference_lookup: ConferenceLookup | None = None,
    ) -> None:
        self._config = config
        self._conference_lookup = conference_lookup
        self._ratings: dict[int, float] = {}
        self._game_counts: dict[int, int] = {}

    # ── Public: core Elo math ────────────────────────────────────────────

    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> float:
        """Logistic expected score for team A against team B.

        ``expected = 1 / (1 + 10^((r_b − r_a) / 400))``
        """
        exponent = (rating_b - rating_a) / 400.0
        return 1.0 / (1.0 + float(10.0**exponent))

    def get_rating(self, team_id: int) -> float:
        """Return current Elo rating for *team_id* (initial_rating if unseen)."""
        return self._ratings.get(team_id, self._config.initial_rating)

    def update_game(  # noqa: PLR0913
        self,
        w_team_id: int,
        l_team_id: int,
        w_score: int,
        l_score: int,
        loc: str,
        is_tournament: bool,
        *,
        num_ot: int = 0,
    ) -> tuple[float, float]:
        """Process one game and update ratings.

        Returns ``(elo_w_before, elo_l_before)`` — the ratings *before* this
        game's update — suitable for use as feature values.

        Parameters
        ----------
        w_team_id, l_team_id
            Winner and loser team IDs.
        w_score, l_score
            Final scores (raw).
        loc
            ``"H"`` (winner home), ``"A"`` (winner away), ``"N"`` (neutral).
        is_tournament
            Whether this is a tournament game.
        num_ot
            Number of overtime periods (used for margin rescaling).
        """
        r_w = self.get_rating(w_team_id)
        r_l = self.get_rating(l_team_id)

        # Snapshot before-ratings for feature use
        elo_w_before = r_w
        elo_l_before = r_l

        # Apply home-court adjustment to effective ratings for expected calc
        eff_r_w = r_w
        eff_r_l = r_l
        if loc == "H":
            # Winner is home — deflate their effective rating
            eff_r_w -= self._config.home_advantage_elo
        elif loc == "A":
            # Winner is away → loser is home — deflate loser's effective rating
            eff_r_l -= self._config.home_advantage_elo

        expected_w = self.expected_score(eff_r_w, eff_r_l)
        expected_l = 1.0 - expected_w

        # Rescale OT scores for margin calculation
        adj_w = rescale_overtime(w_score, num_ot)
        adj_l = rescale_overtime(l_score, num_ot)
        margin = int(round(adj_w - adj_l))

        mult = self._margin_multiplier(margin)

        k_w = self._effective_k(w_team_id, is_tournament)
        k_l = self._effective_k(l_team_id, is_tournament)

        k_eff_w = k_w * mult
        k_eff_l = k_l * mult

        # actual: 1.0 for winner, 0.0 for loser
        self._ratings[w_team_id] = r_w + k_eff_w * (1.0 - expected_w)
        self._ratings[l_team_id] = r_l + k_eff_l * (0.0 - expected_l)

        # Increment game counts
        self._game_counts[w_team_id] = self._game_counts.get(w_team_id, 0) + 1
        self._game_counts[l_team_id] = self._game_counts.get(l_team_id, 0) + 1

        return (elo_w_before, elo_l_before)

    # ── Public: season management ────────────────────────────────────────

    def apply_season_mean_reversion(self, season: int) -> None:
        """Regress each team toward its conference mean (or global mean).

        No-op when no prior ratings exist or no ``ConferenceLookup`` was
        provided (falls back to global mean for all teams).
        """
        if not self._ratings:
            return

        fraction = self._config.mean_reversion_fraction
        global_mean = sum(self._ratings.values()) / len(self._ratings)

        if self._conference_lookup is None:
            # Regress all toward global mean
            for tid in self._ratings:
                self._ratings[tid] = self._ratings[tid] + fraction * (global_mean - self._ratings[tid])
            return

        # Group teams by conference
        conf_teams: dict[str, list[int]] = {}
        no_conf: list[int] = []
        for tid in self._ratings:
            conf = self._conference_lookup.get(season, tid)
            if conf is not None:
                conf_teams.setdefault(conf, []).append(tid)
            else:
                no_conf.append(tid)

        # Compute conference means
        conf_means: dict[str, float] = {}
        for conf, tids in conf_teams.items():
            conf_means[conf] = sum(self._ratings[t] for t in tids) / len(tids)

        # Regress toward conference mean
        for conf, tids in conf_teams.items():
            cm = conf_means[conf]
            for tid in tids:
                self._ratings[tid] = self._ratings[tid] + fraction * (cm - self._ratings[tid])

        # Teams without conference info: regress toward global mean
        for tid in no_conf:
            self._ratings[tid] = self._ratings[tid] + fraction * (global_mean - self._ratings[tid])

    def reset_game_counts(self) -> None:
        """Reset per-team game counts for a new season (affects variable K)."""
        self._game_counts.clear()

    def start_new_season(self, season: int) -> None:
        """Orchestrate season transition: mean-reversion then reset counts."""
        self.apply_season_mean_reversion(season)
        self.reset_game_counts()

    # ── Public: snapshot / bulk ───────────────────────────────────────────

    def get_all_ratings(self) -> dict[int, float]:
        """Return a copy of the current ratings dict."""
        return dict(self._ratings)

    def process_season(self, games: list[Game], season: int) -> pd.DataFrame:
        """Process all games for a season, returning before-ratings per game.

        Calls ``start_new_season(season)`` if prior ratings exist (i.e., this
        is not the very first season).

        Parameters
        ----------
        games
            Games sorted in chronological order.
        season
            Season year.

        Returns
        -------
        pd.DataFrame
            Columns: ``[game_id, elo_w_before, elo_l_before]``.
        """
        if not games:
            return pd.DataFrame(columns=["game_id", "elo_w_before", "elo_l_before"])

        if self._ratings:
            self.start_new_season(season)

        rows: list[dict[str, object]] = []
        for game in games:
            elo_w, elo_l = self.update_game(
                w_team_id=game.w_team_id,
                l_team_id=game.l_team_id,
                w_score=game.w_score,
                l_score=game.l_score,
                loc=game.loc,
                is_tournament=game.is_tournament,
                num_ot=game.num_ot,
            )
            rows.append(
                {
                    "game_id": game.game_id,
                    "elo_w_before": elo_w,
                    "elo_l_before": elo_l,
                }
            )

        return pd.DataFrame(rows)

    # ── Private helpers ──────────────────────────────────────────────────

    def _effective_k(self, team_id: int, is_tournament: bool) -> float:
        """Determine K-factor based on game count and tournament flag."""
        if is_tournament:
            return self._config.k_tournament
        game_count = self._game_counts.get(team_id, 0)
        if game_count < self._config.early_game_threshold:
            return self._config.k_early
        return self._config.k_regular

    def _margin_multiplier(self, margin: int) -> float:
        """Compute margin-of-victory multiplier: ``min(margin, max)^exponent``."""
        capped = min(abs(margin), self._config.max_margin)
        return float(capped**self._config.margin_exponent)
