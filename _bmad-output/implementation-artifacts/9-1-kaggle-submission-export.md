# Story 9.1: Kaggle Submission Export

Status: ready-for-dev

## Story

As a **data scientist**,
I want to **export my model's predictions in Kaggle March Machine Learning Mania submission format**,
So that **I can submit my bracket predictions directly to the Kaggle competition**.

## Acceptance Criteria

1. **Given** a trained model's probability matrix is available
   **When** the user clicks "Export Kaggle Submission" in the dashboard Presentation page (or runs a CLI command)
   **Then** a CSV file is generated with columns `ID` and `Pred`

2. **And** the CSV covers all pairwise matchups for men's D1 teams in the target season (C(n,2) rows where n = number of teams with games that season — typically ~364 teams yielding ~66K rows)

3. **And** the `ID` column uses the Kaggle format `YYYY_TeamID1_TeamID2` (lower Kaggle team ID first)

4. **And** the `Pred` column contains the model's win probability for TeamID1 (the lower-ID team)

5. **And** the file conforms to the Kaggle `SampleSubmission*.csv` schema (see `data/kaggle/SampleSubmissionStage2.csv` for reference)

6. **And** the export is available via both a dashboard download button and a CLI command

### AC Correction Note

The epic states "2,278 possible team matchups" — this is incorrect. The Kaggle Stage 2 submission format requires pairwise matchups for **all** men's D1 teams (~364 teams → C(364,2) = 66,066 matchups), not just the 68 tournament-field teams. Verified against `data/kaggle/SampleSubmissionStage2.csv` which contains 131,407 rows (66,066 men's + 65,341 women's). This project covers men's only, so the output is ~66K rows.

## Tasks / Subtasks

- [ ] Task 1: Core export function (AC: #1, #2, #3, #4, #5)
  - [ ] 1.1 Create `src/ncaa_eval/evaluation/kaggle_export.py` with pure function `format_kaggle_submission(season, team_ids, prob_matrix) -> str`
  - [ ] 1.2 Generate all C(n,2) pairwise `YYYY_LowerID_HigherID` rows
  - [ ] 1.3 Look up P(lower_id beats higher_id) from the probability matrix
  - [ ] 1.4 Return CSV string with header `ID,Pred`

- [ ] Task 2: CLI command (AC: #6)
  - [ ] 2.1 Add `export` command to `src/ncaa_eval/cli/main.py` via Typer
  - [ ] 2.2 Options: `--run-id` (required), `--season` (required), `--data-dir`, `--output` (default stdout)
  - [ ] 2.3 Load model from `RunStore`, build probability matrix, call `format_kaggle_submission`, write output

- [ ] Task 3: Dashboard integration (AC: #6)
  - [ ] 3.1 Add "Export Kaggle Submission" `st.download_button` to `dashboard/pages/2_Presentation.py`
  - [ ] 3.2 Wire button to `format_kaggle_submission` using the existing `sim_data.prob_matrix` and bracket team data

- [ ] Task 4: Unit tests (AC: all)
  - [ ] 4.1 Test `format_kaggle_submission` with a small synthetic matrix (e.g., 4 teams → 6 rows)
  - [ ] 4.2 Verify CSV header, ID format (lower ID first), probability values, row count = C(n,2)
  - [ ] 4.3 Test CLI `export` command end-to-end (mock RunStore)
  - [ ] 4.4 Test edge case: team ID ordering correctness for all pairs

## Dev Notes

### Architecture & Module Placement

- **New pure function module:** `src/ncaa_eval/evaluation/kaggle_export.py` — follows the existing evaluation subpackage pattern alongside `providers.py`, `bracket.py`, `simulation.py`
- The core export function must be **pure** (no I/O) — takes data in, returns CSV string. This follows the PEP 20 / SOLID gates from Story 8.9.
- CLI and dashboard are thin wrappers that call the core function.

### Kaggle Submission Format

Reference file: `data/kaggle/SampleSubmissionStage2.csv`
```csv
ID,Pred
2025_1101_1102,0.5
2025_1101_1103,0.5
```

- **ID column:** `YYYY_TeamID1_TeamID2` where TeamID1 < TeamID2 (lower Kaggle integer ID first)
- **Pred column:** Win probability for TeamID1 (the lower-ID team)
- **Rows:** All C(n,2) pairwise combinations of men's D1 teams for the given season
- The Kaggle sample files in `data/kaggle/` include both men's (ID < 3000) and women's (ID >= 3000). This project exports men's only.

### Probability Matrix Construction

The model's probability matrix is n×n where entry `P[i,j]` = P(team_i beats team_j). This matrix is already built by `build_probability_matrix()` in `src/ncaa_eval/evaluation/providers.py:165-193`. The export function maps matrix indices to Kaggle team IDs:

- If `team_ids[i] < team_ids[j]`: output `Pred = P[i,j]`
- If `team_ids[i] > team_ids[j]`: output `Pred = 1 - P[i,j]` (complementarity)

**Critical:** The matrix diagonal is zero (team vs. itself) — skip these pairs. The Kaggle format also excludes self-matchups.

### Team ID Source

For Kaggle submission, we need **all** men's D1 team IDs that played games in the target season — not just the 64/68 tournament teams. Use `Repository.get_teams()` from `src/ncaa_eval/ingest/repository.py:32` or filter the games table for teams that played in the season. The tournament bracket's `BracketStructure.team_ids` contains only 64 teams — **do not use this** for Kaggle submission.

### Probability for Non-Tournament Teams

The trained model must provide probabilities for ALL team pairs, not just tournament teams. For stateless models (XGBoost, LogReg), this requires building a feature matrix for all team pairs and calling `model.predict_proba()`. For stateful models (Elo), use `EloProvider.matchup_probability()` for each pair.

The CLI `export` command must:
1. Load the model via `RunStore.load_model(run_id)`
2. Determine model type (stateful vs stateless)
3. For stateful: wrap in `EloProvider`, call `build_probability_matrix(provider, all_team_ids, context)`
4. For stateless: reconstruct the feature server and compute predictions for all pairs
5. Call `format_kaggle_submission(season, team_ids, prob_matrix)`

**Simplification option (recommended for v1):** Focus on Elo model support initially since it can predict any matchup without a feature matrix. For stateless models, document that the feature server must be available and defer full stateless support to a follow-up if complex.

### Dashboard Integration

`dashboard/pages/2_Presentation.py` already has:
- `sim_data.prob_matrix` — n×n probability matrix (but only for 64 tournament teams)
- `sim_data.bracket.team_ids` — the 64 tournament team IDs

For the dashboard button, there are two options:
1. **Tournament-only export:** Use the existing 64-team `prob_matrix` — generates C(64,2) = 2,016 matchups. This is **not** a valid Kaggle submission (too few rows).
2. **Full export:** Build a new probability matrix for all ~364 teams — valid Kaggle submission but requires computing ~66K probabilities on button click.

**Recommendation:** Implement option 2 (full export) to produce a valid Kaggle submission. Add a spinner ("Generating Kaggle submission...") since building the full matrix may take a few seconds for Elo. If the model type doesn't support all-pairs prediction, disable the button with an explanatory tooltip.

### Existing Export Code (Do Not Duplicate)

`dashboard/lib/export.py` contains `export_bracket_csv()` which exports bracket picks (63 game rows). This is a **different format** from Kaggle submission:
- Bracket export: 63 rows, columns `game_number,round,team_id,team_name,seed,win_probability`
- Kaggle export: ~66K rows, columns `ID,Pred`

Do not modify the existing bracket export. The new Kaggle export is a separate function in a separate module.

### CLI Design

Follow the existing Typer pattern in `src/ncaa_eval/cli/main.py`:

```python
@app.command()
def export(
    run_id: str = typer.Option(..., "--run-id", help="Model run ID"),
    season: int = typer.Option(..., "--season", help="Target season year"),
    data_dir: Path = typer.Option(Path("data/"), "--data-dir"),
    output: Path | None = typer.Option(None, "--output", help="Output CSV path (default: stdout)"),
) -> None:
```

The heavy lifting (model loading, matrix building) should be in a separate orchestration function, not inline in the command handler — matching the `train` → `run_training` pattern.

### Testing Strategy

- **Unit tests** in `tests/unit/test_kaggle_export.py`:
  - Small synthetic matrix (4 teams) → verify 6 CSV rows with correct IDs and probabilities
  - Verify lower-ID-first ordering for all pairs
  - Verify CSV header exactly matches `ID,Pred`
  - Verify probability values are in [0, 1]
  - Verify complementarity: P(A beats B) in CSV = 1 - P(B beats A) if you query the opposite pair
- **CLI test** in `tests/unit/test_cli.py` or separate file:
  - Mock `RunStore` and model, verify output CSV format
- Type annotations required: `npt.NDArray[np.float64]` for numpy arrays, `Sequence[int]` for team IDs

### Project Structure Notes

- New files: `src/ncaa_eval/evaluation/kaggle_export.py`, `tests/unit/test_kaggle_export.py`
- Modified files: `src/ncaa_eval/cli/main.py` (add `export` command), `dashboard/pages/2_Presentation.py` (add download button)
- `from __future__ import annotations` required in all new Python files
- `mypy --strict` applies to all new code

### References

- [Source: `_bmad-output/planning-artifacts/epics.md` — Story 9.1 AC]
- [Source: `_bmad-output/planning-artifacts/po-decision-log-epic8.md` — Item 1.3, Decision A]
- [Source: `data/kaggle/SampleSubmissionStage2.csv` — Kaggle format reference]
- [Source: `src/ncaa_eval/evaluation/providers.py` — ProbabilityProvider, build_probability_matrix]
- [Source: `src/ncaa_eval/evaluation/bracket.py` — BracketStructure (64 teams only)]
- [Source: `src/ncaa_eval/model/tracking.py` — RunStore.load_model, RunStore.load_run]
- [Source: `src/ncaa_eval/cli/main.py` — Typer CLI pattern, train command]
- [Source: `dashboard/lib/export.py` — existing bracket CSV export (different format)]
- [Source: `dashboard/pages/2_Presentation.py` — dashboard integration point]

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
