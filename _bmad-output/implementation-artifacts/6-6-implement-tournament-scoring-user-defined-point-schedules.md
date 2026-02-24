# Story 6.6: Implement Tournament Scoring with User-Defined Point Schedules

Status: ready-for-dev

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a data scientist,
I want to apply configurable point schedules to simulated or actual brackets,
So that I can evaluate model value under different pool scoring rules and optimize my entry strategy.

## Acceptance Criteria

1. **Given** simulated brackets (Story 6.5) or actual tournament results, **When** the developer applies a scoring schedule to bracket results, **Then** built-in scoring schedules are available: Standard (1-2-4-8-16-32), Fibonacci (2-3-5-8-13-21), and Seed-Difference Bonus.

2. Custom scoring schedules can be defined via configuration (dict or callable).

3. "Expected Points" is computed by averaging scores across all N simulated brackets.

4. "Bracket Distribution" shows the score distribution across simulations (percentiles, histogram data).

5. Scoring integrates with the plugin registry for user-defined custom scoring functions.

6. Scoring is covered by unit tests with known bracket fixtures and expected point totals.

## Tasks / Subtasks

- [x] Task 1: Fix SCORING_REGISTRY type and add decorator-based registration (AC: #5)
  - [x] 1.1 Widen `SCORING_REGISTRY` type from `dict[str, type[StandardScoring] | type[FibonacciScoring]]` to `dict[str, type]` or a proper `Callable[[], ScoringRule]` factory — addresses 6.5 review follow-up [AI-Review][LOW]
  - [x] 1.2 Implement `register_scoring(name: str)` decorator (mirror `register_model` from `model/registry.py`) and `get_scoring(name: str) -> ScoringRule`, `list_scorings() -> list[str]`
  - [x] 1.3 Register all built-in rules (`StandardScoring`, `FibonacciScoring`, `SeedDiffBonusScoring`) via the decorator
  - [x] 1.4 Unit tests: duplicate registration raises `ValueError`, unknown name raises `ScoringNotFoundError`, `list_scorings` returns sorted names

- [x] Task 2: Implement seed-aware Expected Points computation (AC: #1, #3)
  - [x] 2.1 Implement `compute_expected_points_seed_diff(adv_probs, bracket, P, seed_map) -> ndarray` — per-team EP that includes seed-diff bonus for upset wins, computed analytically (extend Phylourny output with per-matchup seed lookup)
  - [x] 2.2 The function traverses the bracket, at each internal node computing the bonus-weighted expected contribution: for each pair of potential opponents, multiply P(team_i wins that game) × seed_diff_bonus(seed_i, seed_j) × P(opponent_j reaches that game)
  - [x] 2.3 Integrate into `simulate_tournament` orchestrator: when a `SeedDiffBonusScoring` rule is in `scoring_rules`, use the seed-aware EP function instead of `compute_expected_points`
  - [x] 2.4 Unit tests: verify seed-diff EP for known small-bracket fixtures; verify equals standard EP when all seeds are identical; verify converges with MC at large N

- [x] Task 3: Implement `compute_most_likely_bracket` (AC: #4, deferred from 6.5)
  - [x] 3.1 Implement `compute_most_likely_bracket(bracket, P) -> tuple[ndarray, float]` — returns `(winners, log_likelihood)` where `winners` is shape `(n_games,)` array of team indices for the max-likelihood bracket; greedy traversal picking `argmax(P[left, right])` at each game
  - [x] 3.2 Return type is a frozen `MostLikelyBracket` dataclass with fields: `winners: tuple[int, ...]`, `champion_team_id: int`, `log_likelihood: float`
  - [x] 3.3 Unit tests: deterministic matrix yields perfect bracket; uniform matrix yields any valid bracket; log_likelihood is negative sum of log(max(P[i,j], P[j,i]))

- [x] Task 4: Implement `BracketDistribution` analysis (AC: #4)
  - [x] 4.1 Implement `BracketDistribution` frozen dataclass with fields: `scores: ndarray` (raw per-sim scores), `percentiles: dict[int, float]` (5, 25, 50, 75, 95), `mean: float`, `std: float`, `histogram_bins: ndarray`, `histogram_counts: ndarray`
  - [x] 4.2 Implement `compute_bracket_distribution(score_distribution: ndarray, n_bins: int = 50) -> BracketDistribution` — computes all distribution statistics from raw MC scores
  - [x] 4.3 Integrate into `SimulationResult`: add `bracket_distributions: dict[str, BracketDistribution] | None` field (MC only)
  - [x] 4.4 Update `simulate_tournament_mc` to populate `bracket_distributions` from existing `score_distribution` arrays
  - [x] 4.5 Unit tests: verify percentile ordering, histogram bin count, mean/std against numpy reference, None for analytical path

- [x] Task 5: Implement dict-based custom scoring configuration (AC: #2)
  - [x] 5.1 Implement `DictScoring(ScoringRule)` — wraps a `dict[int, float]` mapping `round_idx → points`; validates exactly 6 entries (rounds 0–5); raises `ValueError` on missing rounds
  - [x] 5.2 Implement `scoring_from_config(config: dict[str, Any]) -> ScoringRule` factory — dispatches on `config["type"]`: `"standard"`, `"fibonacci"`, `"seed_diff_bonus"`, `"dict"` (reads `config["points"]`), `"custom"` (reads `config["callable"]`)
  - [x] 5.3 Unit tests: dict scoring round-trip, config factory for each type, invalid config raises ValueError

- [ ] Task 6: Fix MC score_distribution to use actual per-sim bracket scores (AC: #3, #4)
  - [ ] 6.1 Currently `score_distribution` tracks chalk-bracket scores (whether pre-game favorite won). Refactor MC engine to track the actual per-sim bracket outcome and score it: for each sim, the winners array records who won each game; score = sum of `points_per_round(r)` for each correct pick (if the sim's winner matches the scoring bracket's pick). For "Expected Points" style scoring, the existing `adv_probs @ points` path is correct. But for score_distribution, each sim should record its own total score = sum of points for all 63 games in that sim.
  - [ ] 6.2 For standard scoring (no seed-diff), total_score_per_sim = sum over rounds of `n_correct_picks_in_round × points_per_round` — but in MC each sim IS a complete bracket, so every game has a winner and every winner earns points. The per-sim score is simply `sum(points_per_round(r) × n_games_in_round for r in range(6))` which is constant (192 for standard). The DISTRIBUTION comes from: "if I pick this bracket as my entry, how many of these simulated realities would give me what score?" This requires comparing a CHOSEN bracket against each simulation.
  - [ ] 6.3 Implement `score_bracket_against_sims(chosen_bracket: ndarray, sim_winners: ndarray, scoring_rules: Sequence[ScoringRule]) -> dict[str, ndarray]` — for each sim, count how many of the chosen bracket's picks match the sim's outcomes, weighted by round points. Returns per-sim score array.
  - [ ] 6.4 Update `simulate_tournament_mc` to store `sim_winners` (shape `(n_simulations, 63)`) in `SimulationResult` for downstream bracket-vs-sim scoring
  - [ ] 6.5 Unit tests: deterministic matrix → chosen bracket matches all sims → max score; verify score distribution has genuine variance when bracket picks differ from sim outcomes

- [ ] Task 7: Export new public API and verify quality (AC: all)
  - [ ] 7.1 Add new types and functions to `evaluation/__init__.py` `__all__`: `BracketDistribution`, `MostLikelyBracket`, `DictScoring`, `register_scoring`, `get_scoring`, `list_scorings`, `ScoringNotFoundError`, `compute_expected_points_seed_diff`, `compute_most_likely_bracket`, `compute_bracket_distribution`, `score_bracket_against_sims`, `scoring_from_config`
  - [ ] 7.2 Verify `mypy --strict` passes on all source files
  - [ ] 7.3 Verify `ruff check` passes
  - [ ] 7.4 Run full test suite — all existing 658+ tests plus new tests pass

## Dev Notes

### Story 6.5 Review Follow-ups to Address

Two action items from Story 6.5 code review must be resolved in this story:

1. **[AI-Review][MEDIUM]** AC #4 partially missing: "most likely bracket (max likelihood)" not in `SimulationResult` and no `compute_most_likely_bracket()` function → **Task 3**
2. **[AI-Review][LOW]** `SCORING_REGISTRY` type annotation is overly narrow (`dict[str, type[StandardScoring] | type[FibonacciScoring]]`) — limits registry extensibility → **Task 1**

### Existing Code — DO NOT Reimplement

All of the following exist in `src/ncaa_eval/evaluation/simulation.py` (Story 6.5):

- `BracketNode`, `BracketStructure`, `build_bracket` — bracket data structures (DO NOT MODIFY)
- `MatchupContext` — matchup context frozen dataclass (DO NOT MODIFY)
- `ProbabilityProvider`, `MatrixProvider`, `EloProvider`, `build_probability_matrix` — probability provider protocol and implementations (DO NOT MODIFY)
- `compute_advancement_probs` — Phylourny analytical computation (DO NOT MODIFY)
- `compute_expected_points` — standard matrix-vector EP (DO NOT MODIFY, extend with seed-aware variant)
- `simulate_tournament_mc` — vectorized MC engine (MODIFY to add sim_winners tracking and bracket_distributions)
- `simulate_tournament` — orchestrator (MODIFY to handle SeedDiffBonusScoring dispatch and bracket_distributions)
- `ScoringRule` protocol, `StandardScoring`, `FibonacciScoring`, `SeedDiffBonusScoring`, `CustomScoring` — scoring rules (MODIFY registry, keep rule implementations)
- `SCORING_REGISTRY` — replace with decorator-based registry
- `SimulationResult` — add `bracket_distributions` and `sim_winners` fields
- `_collect_leaves` — internal helper (DO NOT MODIFY)

Existing code in other modules (DO NOT MODIFY):
- `evaluation/metrics.py`: metric functions
- `evaluation/splitter.py`: walk_forward_splits, CVFold
- `evaluation/backtest.py`: run_backtest, FoldResult, BacktestResult
- `model/registry.py`: `register_model`, `get_model`, `list_models` — **PATTERN REFERENCE for scoring registry**
- `transform/normalization.py`: `TourneySeed`, `TourneySeedTable`, `parse_seed`

### Scoring Rule Architecture

**ScoringRule Protocol** (existing, keep as-is):
```python
@runtime_checkable
class ScoringRule(Protocol):
    @property
    def name(self) -> str: ...
    def points_per_round(self, round_idx: int) -> float: ...
```

**New decorator registry** (Task 1) — mirror `model/registry.py`:
```python
_SCORING_REGISTRY: dict[str, type] = {}

def register_scoring(name: str) -> Callable[[_T], _T]: ...
def get_scoring(name: str) -> type: ...
def list_scorings() -> list[str]: ...
class ScoringNotFoundError(KeyError): ...
```

Remove the module-level `SCORING_REGISTRY` dict and replace with the decorator-based approach. Existing tests referencing `SCORING_REGISTRY` must be updated.

### Seed-Diff EP Algorithm

The standard EP computation is `adv_probs @ points_vector` (dot product). This works because `points_per_round` is constant per round.

For `SeedDiffBonusScoring`, the bonus depends on the specific matchup (who beats whom), not just the round. The analytical computation requires:

For each internal node at round r:
- For each team i that could win at this node:
  - For each possible opponent j:
    - `EP_bonus[i] += P(team_i wins at node) × P(team_j is the opponent) × seed_diff_bonus(seed_i, seed_j)`

This can be computed efficiently from the WPV (Win Probability Vector) data already available from `compute_advancement_probs`. Each internal node's WPV tells us the probability that each team reaches (and wins) that node. The left/right child WPVs tell us who the possible opponents are.

Implementation: post-order traversal of the bracket tree (same as `compute_advancement_probs`), at each internal node computing:
```python
bonus_ep[i] += sum_j(left_wpv[i] * P[i,j] * right_wpv[j] * bonus(seed_i, seed_j))
             + sum_j(right_wpv[i] * P[i,j] * left_wpv[j] * bonus(seed_i, seed_j))
```

### MC Score Distribution Semantics

**Current implementation (Story 6.5)**: `score_distribution` tracks "chalk score" — how many pre-game favorites won per simulation, weighted by round points. This creates variance but doesn't answer the user's actual question.

**What users want**: "If I submit bracket X to my pool, what's the distribution of points I'd score across all possible tournament realities?"

This requires:
1. A **chosen bracket** (e.g., the most-likely bracket, or a user-selected bracket)
2. Comparing that bracket against each of the N simulated realities
3. For each sim, counting how many of the chosen bracket's picks match the sim's actual outcomes

**New `sim_winners` field**: Store the full (n_simulations, 63) array of game winners from MC. This is the raw data needed for any bracket-vs-reality scoring analysis. Memory: 63 × N × 4 bytes = ~2.5 MB for N=10,000.

### `SimulationResult` Field Changes

Add to existing frozen dataclass:
- `bracket_distributions: dict[str, BracketDistribution] | None` — distribution statistics per scoring rule (MC only)
- `sim_winners: npt.NDArray[np.int32] | None` — shape `(n_simulations, 63)` array of game winners (MC only; `None` for analytical)

`SimulationResult` is frozen, so adding fields is safe (no mutation). Existing code that constructs `SimulationResult` (in `simulate_tournament_mc` and `simulate_tournament`) must be updated to include the new fields.

### File Placement

Modified files:
- `src/ncaa_eval/evaluation/simulation.py` — all changes (registry, seed-diff EP, most-likely bracket, bracket distribution, MC updates, new dataclasses)
- `src/ncaa_eval/evaluation/__init__.py` — add new public exports
- `tests/unit/test_evaluation_simulation.py` — new tests for all new functionality

No new source files needed — all scoring functionality belongs in the existing `simulation.py` module.

### Project Structure Notes

```
src/ncaa_eval/evaluation/
  __init__.py      # existing — add new exports
  backtest.py      # existing — NO CHANGES
  metrics.py       # existing — NO CHANGES
  simulation.py    # existing — MODIFY (registry, seed-diff EP, most-likely bracket, distributions, MC)
  splitter.py      # existing — NO CHANGES
```

### Architecture Constraints

- **NFR1 (Vectorization)**: Seed-diff EP computation must use matrix operations, not per-team Python loops. `compute_most_likely_bracket` is a single O(n_games) traversal — acceptable. `compute_bracket_distribution` uses numpy percentile/histogram.
- **NFR3 (Extensibility)**: Decorator-based scoring registry allows external users to register custom scoring rules without modifying core code.
- **mypy --strict**: All types must be fully annotated. Use `npt.NDArray[np.float64]` for array types.
- **`from __future__ import annotations`**: Required in all files.
- **Google docstring style**: `Args:`, `Returns:`, `Raises:`.
- **Frozen dataclasses**: For all new immutable containers (`BracketDistribution`, `MostLikelyBracket`).

### Dependencies

No new dependencies. Everything uses:
- `numpy` (already in pyproject.toml) — matrix ops, statistics
- `logging` (stdlib) — progress reporting
- Standard library `dataclasses`, `typing` — data structures

### Previous Story Learnings (Stories 6.1–6.5)

- **Google docstring style**: NOT NumPy style. `Args:`, `Returns:`, `Raises:`.
- **Frozen dataclasses**: All result containers must be frozen. Pattern: `FoldResult`, `BacktestResult`, `SimulationResult`.
- **Protocol type contracts**: Return types precisely annotated — `npt.NDArray[np.float64]` shape comments, not just `np.ndarray`.
- **Mathematical claim verification**: Always verify formulas against known fixtures. Story 6.5 caught wrong formula notation (P vs P^T) and wrong Fibonacci total (164 vs 231).
- **Registry pattern**: Follow `model/registry.py` — decorator-based with `_REGISTRY` dict, `register_X`, `get_X`, `list_Xs`, `XNotFoundError`. Prevents duplicate registration.
- **Chalk-bracket score_distribution**: Current implementation has genuine variance but doesn't answer "how would MY bracket score?" — this story fixes that.
- **`_collect_leaves` helper**: Returns leaf team indices in bracket order. Used by MC engine.
- **`N_ROUNDS = 6`, `N_GAMES = 63`**: Module constants. Don't redefine.
- **`_REGION_SEED_ORDER` and `_REGION_ORDER`**: Hardcoded NCAA bracket structure. Don't modify.

### Performance Targets

| Operation | Target |
|:---|:---|
| Seed-diff EP (64 teams, analytical) | < 5 ms |
| Most-likely bracket (64 teams) | < 1 ms |
| Bracket distribution stats (N=10,000) | < 10 ms |
| Score-bracket-against-sims (N=10,000) | < 50 ms |
| MC simulation with sim_winners (N=10,000) | < 3 s |

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 6.6 — acceptance criteria]
- [Source: _bmad-output/implementation-artifacts/6-5-implement-monte-carlo-tournament-simulator.md — previous story learnings, review follow-ups]
- [Source: specs/research/tournament-simulation-confidence.md — spike research for scoring and simulation design]
- [Source: specs/03-prd.md — FR7 Tournament Scoring]
- [Source: specs/05-architecture-fullstack.md §4.1 — TournamentBracket entity]
- [Source: src/ncaa_eval/evaluation/simulation.py — existing implementation to extend]
- [Source: src/ncaa_eval/model/registry.py — decorator registry pattern reference]
- [Source: src/ncaa_eval/evaluation/__init__.py — existing exports to extend]

## Dev Agent Record

### Agent Model Used

{{agent_model_name_version}}

### Debug Log References

### Completion Notes List

### File List
