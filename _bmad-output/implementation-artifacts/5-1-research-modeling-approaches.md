# Story 5.1: Research Modeling Approaches

Status: review

## Story

As a data scientist,
I want a documented survey of modeling approaches used for NCAA tournament prediction,
So that I can ensure the Model ABC supports all viable approaches and select the best reference implementations.

## Acceptance Criteria

1. **Kaggle MMLM Community Review** — Review documented community solutions and published writeups for Kaggle March Machine Learning Mania across multiple competition years (2014–2025). Secondary sources (Medium, mlcontests.com, public GitHub repos) are the practical approach since Kaggle discussion boards require authentication.

2. **Stateful Models Catalogued** — Stateful model approaches are catalogued: Elo variants (standard, margin-scaled, variable-K), Glicko-2, TrueSkill, and any custom rating systems found in top solutions.

3. **Stateless Models Catalogued** — Stateless model approaches are catalogued: XGBoost, LightGBM, CatBoost, logistic regression, neural networks (LSTM, Transformer), random forest, SVM, and ensemble methods found in competition solutions.

4. **Hybrid Approaches Documented** — Hybrid approaches are documented (e.g., Elo features fed into XGBoost, ordinal composites as features for gradient boosting, stacking/blending of model outputs).

5. **Model ABC Requirements Derived** — Requirements for the Model ABC are derived from the survey: what interface must support all catalogued approaches (stateful per-game `update`, stateless batch `train`/`predict`, `save`/`load` persistence, hyperparameter configuration, plugin registration).

6. **Reference Models Recommended** — Reference models to implement first (Stories 5.3 and 5.4) are recommended with rationale, including which approaches are most proven for NCAA prediction and most useful as templates.

7. **PO Decision Gate** — The product owner reviews the spike findings and approves which modeling approaches to include in the MVP scope before downstream stories (5.2–5.5) begin implementation.

8. **Post-PO SM Downstream Update** — After PO approval of the spike findings, the SM updates the downstream story descriptions in `epics.md` to incorporate all building blocks and scope decisions from the research document — adding new story placeholders as needed and moving deferred items to the Post-MVP Backlog.

9. **Research Document Committed** — The findings are committed as `specs/research/modeling-approaches.md`.

## Tasks / Subtasks

- [x] Task 1: Survey Kaggle MMLM solutions and published writeups (AC: #1, #3, #4)
  - [x] 1.1 Search for top solution writeups from 2014–2025 competitions (Medium, mlcontests.com, GitHub, personal blogs)
  - [x] 1.2 Document model types used by top finishers: what architectures, what features, what calibration methods
  - [x] 1.3 Identify recurring patterns across winning solutions (which approaches consistently place well)
  - [x] 1.4 Document any competition-specific quirks (metric changes: LogLoss → Brier, men's+women's combined in 2023)

- [x] Task 2: Catalogue stateful model approaches (AC: #2)
  - [x] 2.1 Document Elo variants: standard, margin-scaled (Silver/SBCB), variable-K, home-court adjusted
  - [x] 2.2 Document Glicko-2: RD (Rating Deviation) and volatility parameters, uncertainty quantification
  - [x] 2.3 Document TrueSkill: factor graph approach, Gaussian belief propagation
  - [x] 2.4 Document any novel custom rating systems found in community solutions
  - [x] 2.5 Compare: what distinct signal does each provide beyond Elo? (reference Post-MVP Backlog items for TrueSkill/Glicko-2)

- [x] Task 3: Catalogue stateless model approaches (AC: #3)
  - [x] 3.1 Document XGBoost: hyperparameter ranges, feature importance, calibration approaches used in NCAA context
  - [x] 3.2 Document LightGBM/CatBoost: when preferred over XGBoost, performance comparisons
  - [x] 3.3 Document logistic regression: Bayesian logistic regression (Landgraf 2017 winner), regularized variants
  - [x] 3.4 Document neural network approaches: LSTM (temporal game sequences), Transformer architectures
  - [x] 3.5 Document ensemble/stacking: how top solutions combine multiple model outputs

- [x] Task 4: Document hybrid approaches (AC: #4)
  - [x] 4.1 Document Elo-as-feature pattern (already implemented as Story 4.8 building block → Story 5.3 model wrapper → XGBoost in 5.4)
  - [x] 4.2 Document ordinal composite features + gradient boosting (maze508 2023 gold: top-10 rating systems → XGBoost)
  - [x] 4.3 Document stacking/blending architectures (model outputs as meta-features)
  - [x] 4.4 Document game-theory/meta-modeling (Landgraf 2017: modeling competitors' submissions)

- [x] Task 5: Derive Model ABC interface requirements (AC: #5)
  - [x] 5.1 Define the minimal interface that supports ALL catalogued approaches
  - [x] 5.2 Distinguish stateful vs. stateless contract requirements
  - [x] 5.3 Define persistence requirements (`save`/`load` for model state + hyperparameters)
  - [x] 5.4 Define plugin-registry requirements (register by name, runtime discovery)
  - [x] 5.5 Define hyperparameter configuration schema (Pydantic validation for config dicts)
  - [x] 5.6 Define prediction output contract (calibrated probabilities, not raw scores)
  - [x] 5.7 Consider scikit-learn Estimator/Predictor interface compatibility

- [x] Task 6: Recommend reference implementations (AC: #6)
  - [x] 6.1 Recommend stateful reference model (Elo) with rationale (proven baseline, reuses EloFeatureEngine from Story 4.8)
  - [x] 6.2 Recommend stateless reference model (XGBoost) with rationale (most successful in competition, tabular data standard)
  - [x] 6.3 Assess whether LightGBM or neural net should be a third reference model or deferred to Post-MVP
  - [x] 6.4 Document recommended hyperparameter ranges for each reference model

- [x] Task 7: Write research document and commit (AC: #7, #8, #9)
  - [x] 7.1 Write `specs/research/modeling-approaches.md` with all findings
  - [x] 7.2 Include a "Scope Recommendation" section with options for PO decision
  - [x] 7.3 Include a "Model ABC Interface Specification" section for Story 5.2 consumption
  - [x] 7.4 Include an "Equivalence Groups" analysis (which models are redundant vs. distinct)
  - [x] 7.5 Commit research document
  - [x] 7.6 Update sprint-status.yaml: `5-1-research-modeling-approaches` → `review`

- [ ] Task 8: Post-PO SM downstream update (AC: #8) — SM WORK, NOT DEV WORK
  - [ ] 8.1 After PO approves spike findings, SM updates downstream story descriptions (Stories 5.2–5.5) in `_bmad-output/planning-artifacts/epics.md` to reflect all scope decisions and building blocks from `specs/research/modeling-approaches.md`
  - [ ] 8.2 SM adds new story placeholders in `epics.md` as needed (e.g., logistic regression in Story 5.4 if PO selects Option B or C)
  - [ ] 8.3 SM moves deferred items (LightGBM, Glicko-2, TrueSkill, LSTM) to the Post-MVP Backlog section in `epics.md`
  - [ ] 8.4 SM updates sprint-status.yaml: `5-1-research-modeling-approaches` → `done` (only after all AC 8 work complete)

## Dev Notes

### Architecture & Design Constraints

- **Spike story**: This is a research-only story — NO production code is written. Output is a Markdown research document at `specs/research/modeling-approaches.md`.
- **Research output path**: `specs/research/` (NOT `docs/`). Research spikes are planning artifacts, not Sphinx documentation.
- **PO decision gate required**: Do NOT commit scope decisions to `epics.md` — present recommendations with trade-offs and let the PO decide. This was a critical lesson from Story 2.1 (see template-requirements.md).
- **Post-PO SM work**: After PO approval, the SM updates downstream Stories 5.2–5.5 in `epics.md`. This is SM work, not dev agent work.

### Existing Codebase Context (Critical — DO NOT Reimplement)

The project already has a comprehensive feature engineering pipeline (Epic 4) that models will consume:

| Building Block | Module | Key API | Story |
|:---|:---|:---|:---|
| Feature serving | `transform.feature_serving` | `StatefulFeatureServer.serve_season_features(year, mode)` → DataFrame | 4.7 |
| Elo feature engine | `transform.elo` | `EloFeatureEngine.update_game()`, `.process_season()` | 4.8 |
| Chronological serving | `transform.serving` | `ChronologicalDataServer.get_chronological_season(year)` | 4.2 |
| Batch ratings | `transform.opponent` | `compute_srs_ratings()`, `compute_ridge_ratings()`, `compute_colley_ratings()` | 4.6 |
| Graph features | `transform.graph` | `compute_pagerank()`, `compute_betweenness_centrality()` | 4.5 |
| Sequential features | `transform.sequential` | `SequentialTransformer` — rolling, EWMA, momentum, streak, Four Factors | 4.4 |
| Normalization | `transform.normalization` | `MasseyOrdinalsStore`, `TourneySeedTable`, `ConferenceLookup` | 4.3 |
| Calibration | `transform.calibration` | `IsotonicCalibrator`, `SigmoidCalibrator` | 4.7 |
| Data layer | `ingest.schema` | `Game` (Pydantic v2), `Team`, `Season` | 2.2 |
| Repository | `ingest.repository` | `ParquetRepository` (abstract `Repository` ABC) | 2.2 |

**Feature serving output columns** (from `StatefulFeatureServer`):
- Metadata: `game_id`, `season`, `day_num`, `date`, `team_a_id`, `team_b_id`, `is_tournament`, `loc_encoding`, `team_a_won`
- Feature blocks: sequential stats (rolling, EWMA, momentum), graph centrality (PageRank, betweenness), batch ratings (SRS, Ridge, Colley), ordinal composites, seed info, Elo
- Matchup deltas: `seed_diff`, `delta_ordinal_composite`, `delta_srs`, `delta_ridge`, `delta_colley`, `delta_elo`

**Placeholder directories already exist**:
- `src/ncaa_eval/model/__init__.py` — EMPTY (Story 5.2 will populate)
- `src/ncaa_eval/evaluation/__init__.py` — EMPTY (Story 6 will populate)

### Feature Engineering Research Already Done (Story 4.1)

The feature engineering techniques research (`specs/research/feature-engineering-techniques.md`) already covers:
- Opponent adjustment equivalence groups (SRS ≈ Massey ≈ Ridge as margin-adjusted batch ratings; Colley distinct; Elo distinct)
- Community techniques from Kaggle MMLM (Section 6)
- Library building blocks catalog mapping to Stories 4.2–4.8 (Section 7)
- 538-based approach dominance 2018–2023
- `goto_conversion` for calibration assessment

**This spike should NOT re-survey feature engineering** — focus on the MODEL layer: how features are consumed, what model architectures work, and what the Model ABC interface should look like.

### Web Research Findings (Pre-loaded Context)

These findings from web research should be used as starting points — verify and expand during the spike:

**Kaggle MMLM Top Solution Patterns (2014–2025):**
- **2017 winner (Landgraf)**: Bayesian logistic regression with team efficiency ratings + distance-from-home. Modeled competitors' submissions via mixed-effects model. Highlights: simplicity + meta-strategy can win.
- **2023 gold (maze508)**: Top-10 external rating systems (Pomeroy, Moore, Sagarin) + win rates + box-score aggregates → XGBoost with Recursive Feature Elimination. Changed metric: Brier Score (2023+), men's+women's combined.
- **2024 winner**: R-based Monte Carlo simulation using third-party ratings + personal intuitions. Not a pure ML approach.
- **2025**: XGBoost outperformed CatBoost and LightGBM per winner.
- **Recurring pattern**: Elo/rating-based features + gradient boosting = most consistent approach across years.

**Deep Learning NCAA Research (2025 arXiv 2508.02725):**
- LSTM + Transformer comparison for NCAA tournament prediction
- Transformer with BCE: highest AUC (0.8473) but worse calibration
- LSTM with Brier loss: superior probability calibration
- Features: GLM team quality metrics, Elo ratings, seed differences, box-score stats

**Model Performance Benchmarks (from literature):**
- XGBoost: ~90% test accuracy on regular-season games (but tournament prediction is harder)
- Logistic regression (regularized): top 99.5% ESPN Challenge
- Neural networks: ~67% accuracy (some studies), but not consistently top in Kaggle

**Scikit-learn Interface Conventions:**
- `Estimator` ABC: must implement `fit(X, y)` (equivalent to our `train`)
- `Predictor` ABC: must implement `predict(X)` (equivalent to our `predict`)
- Model serialization: `joblib.dump()`/`joblib.load()` preferred over pickle for numpy arrays
- Pydantic + ABC pattern: define reusable interfaces with data validation

### Previous Story Learnings (Critical)

**From Story 4.8 (Elo Feature Building Block):**
- `EloFeatureEngine` is a FEATURE BUILDING BLOCK, not a model. Story 5.3 wraps it as a Model ABC plugin with `train`/`predict`/`save`. The research doc must clearly distinguish:
  - **Feature Elo** (4.8): computes ratings to feed as INPUT features to other models
  - **Model Elo** (5.3): uses Elo ratings directly as a PREDICTIVE MODEL (predicts game outcome from rating difference)
- The EloFeatureEngine API: `update_game()` returns before-ratings, `process_season()` for bulk processing, `start_new_season()` for mean-reversion. Model Elo in 5.3 wraps this.

**From Story 4.1 (Feature Engineering Spike):**
- Kaggle discussion boards require authentication — use secondary sources
- Code examples in research docs must be anti-pattern-free (no iterrows, no magic numbers)
- arXiv citations need explicit verification
- Spike sprint-status update should say `→ review` not `→ done`

**From Story 2.1 (Data Source Spike):**
- DO NOT unilaterally commit scope decisions — present recommendations to PO
- Research doc must have a clear "Scope Recommendation" section with options and trade-offs
- Post-PO, SM updates downstream story descriptions in epics.md

### Project Conventions (Must Follow)

- `from __future__ import annotations` required in all Python files (but this is a docs-only spike — no Python files created)
- Conventional commits: `docs(spike): research modeling approaches for Story 5.1`
- Research output: `specs/research/modeling-approaches.md`
- `mypy --strict` and Ruff do NOT apply to Markdown research documents

### Dependencies Already Installed

These are already in `pyproject.toml` and available in the conda env — the research doc should reference them:
- `xgboost` — gradient boosting (Story 5.4)
- `scikit-learn` — logistic regression, model utilities, calibration
- `joblib` — model serialization, parallel execution
- `pydantic` (^2.12.5) — config validation for model hyperparameters

### Key Research Questions to Answer

1. **What model interface supports both stateful (Elo) and stateless (XGBoost)?** — The ABC must handle per-game iteration AND batch training.
2. **Should the ABC mirror scikit-learn's Estimator/Predictor convention?** — Pros: familiarity, potential sklearn pipeline compatibility. Cons: NCAA-specific requirements (walk-forward, tournament vs. regular season) may not fit sklearn patterns.
3. **Is a neural network reference model justified for MVP?** — LSTM/Transformer showed promise but are more complex to train and less proven in competition.
4. **What calibration approach integrates best with the model layer?** — Calibration is already implemented in `transform.calibration` (IsotonicCalibrator, SigmoidCalibrator). Should models output raw probabilities and let the caller calibrate, or should calibration be built into the model?
5. **What persistence format is best for model artifacts?** — joblib (sklearn convention), JSON (hyperparams), Parquet (predictions)?

### Project Structure Notes

- Output file: `specs/research/modeling-approaches.md` (new)
- Modified: `_bmad-output/implementation-artifacts/sprint-status.yaml` (status update)
- No source code files created or modified

### References

- [Source: _bmad-output/planning-artifacts/epics.md — Epic 5 stories, FR6, NFR3]
- [Source: specs/05-architecture-fullstack.md — Model ABC strategy pattern, tech stack, coding standards]
- [Source: specs/research/feature-engineering-techniques.md — Community techniques, equivalence groups, library catalog]
- [Source: _bmad-output/implementation-artifacts/4-8-implement-dynamic-rating-features-elo-feature-building-block.md — Elo feature engine implementation details]
- [Source: src/ncaa_eval/transform/feature_serving.py — StatefulFeatureServer output contract]
- [Source: src/ncaa_eval/transform/elo.py — EloConfig, EloFeatureEngine API]
- [Source: src/ncaa_eval/ingest/repository.py — Repository ABC pattern (existing pattern to follow)]
- [Source: src/ncaa_eval/ingest/connectors/base.py — Connector ABC pattern (existing ABC example)]
- [Source: _bmad-output/planning-artifacts/template-requirements.md — Spike story patterns, PO decision gate, post-PO SM update]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

No debug issues encountered — this is a research-only spike with no production code.

### Completion Notes List

- **Task 1 (Survey):** Compiled year-by-year solution summary for 2014–2025 Kaggle MMLM competitions from secondary sources (Medium writeups, mlcontests.com, personal blogs, public GitHub repos). Identified 5 recurring patterns: feature quality > model complexity, Elo/ratings + GBDT as dominant pattern, calibration as first-class concern, 538 era ending, and competition quirks. Documented model type frequency across top solutions.
- **Task 2 (Stateful Catalogue):** Documented Elo (with all Silver/SBCB parameters already in EloConfig), Glicko-2 (RD + volatility), TrueSkill (factor graphs), LRMC (Markov chains), and mixed-effects models. Assessed each for distinct signal beyond Elo — concluded Glicko-2/TrueSkill provide marginal signal for full-season snapshots; deferred to Post-MVP.
- **Task 3 (Stateless Catalogue):** Documented XGBoost (with competition-validated hyperparameter ranges), LightGBM, CatBoost, logistic regression (standard + Bayesian), neural networks (LSTM/Transformer per arXiv:2508.02725), and random forest/SVM. Key finding: neural nets underperform GBDT on small NCAA tabular data; tree-based models still outperform deep learning on tabular data (NeurIPS 2022 finding).
- **Task 4 (Hybrid Approaches):** Documented Elo-as-feature pattern (already implemented in pipeline), ordinal composites + GBDT (maze508 2023), stacking/blending architectures, and game-theory meta-modeling (Landgraf 2017). The Elo-as-feature → XGBoost pattern is already the project's architecture.
- **Task 5 (Model ABC):** Derived dual-contract ABC design: `StatefulModel` (per-game update) + `StatelessModel` (batch train), both extending a common `Model` parent. Proposed Pydantic-based config schema, decorator-based plugin registry, and JSON/UBJSON persistence. Justified NOT inheriting from sklearn BaseEstimator (walk-forward doesn't fit sklearn Pipeline).
- **Task 6 (Recommendations):** Recommended Elo as stateful reference (Story 5.3) and XGBoost as stateless reference (Story 5.4). Assessed LightGBM, CatBoost, and neural nets as Post-MVP. Documented hyperparameter ranges for both reference models.
- **Task 7 (Document + Commit):** Wrote comprehensive research document at `specs/research/modeling-approaches.md` with 8 sections including Scope Recommendation (3 options for PO), Model ABC Interface Specification (for Story 5.2), and Equivalence Groups analysis.

### Implementation Plan

N/A — research spike; no production code.

### File List

- `specs/research/modeling-approaches.md` (NEW) — comprehensive research document
- `_bmad-output/implementation-artifacts/5-1-research-modeling-approaches.md` (MODIFIED) — story file updated with task completion, Dev Agent Record
- `_bmad-output/implementation-artifacts/sprint-status.yaml` (MODIFIED) — status: ready-for-dev → review

### Senior Developer Review (AI) — Round 1

**Reviewer:** Claude Sonnet 4.6 (adversarial code review)
**Date:** 2026-02-22
**Issues Found:** 2 HIGH, 5 MEDIUM, 3 LOW
**Issues Fixed:** 7 (all HIGH and MEDIUM; LOW issues L2 acknowledged as research gap)
**Git vs Story Discrepancies:** 0

**Summary of fixes applied to `specs/research/modeling-approaches.md`:**
- **H1 (HIGH):** Added `from pathlib import Path` to Section 5.2 interface pseudocode imports — `Path` was used in `save()`/`load()` without being imported. Would cause copy-paste implementation errors in Story 5.2.
- **H2 (HIGH):** Resolved `predict()` abstract method inconsistency in `StatelessModel`. The class inherited `Model.predict(team_a_id, team_b_id)` as abstract but never overrode it, creating a spec defect that would prevent `mypy --strict` compliance in Story 5.2. Added concrete `predict()` that raises `NotImplementedError` with clear guidance to use `predict_proba(X)`.
- **M1 (MEDIUM):** Added Task 8 (post-PO SM downstream update) to story with 4 explicit subtasks. Per template-requirements.md Story 4.1 retrospective, this absence caused Story 4.1's downstream epic update to be missed until discovered after PO approval.
- **M2 (MEDIUM):** Added arXiv:2508.02725 verification caveat per template-requirements.md Research Spike Pattern. AUC values (0.8473 etc.) are from training knowledge, not live-verified.
- **M3 (MEDIUM):** Added `min_child_weight: int = 3` to `XGBoostModelConfig` pseudocode in Section 5.5 to align with Section 6.4 hyperparameter table. Parameter was present in the recommendation table but absent from the Pydantic config spec.
- **M5 (MEDIUM):** Added `text` language tag to stacking architecture fenced code block for proper rendering.
- **M6 (MEDIUM):** Added concrete evaluation pipeline dispatch pattern to Section 5.3 showing how Epic 6 callers handle both model types polymorphically (`isinstance` dispatch → `predict()` for stateful, `predict_proba(X)` for stateless).
- **L1 (LOW):** Added `https://` prefix to all bare URLs in References section.
- **L3 (LOW):** Added 2024+ FiveThirtyEight discontinuation row to Section 1.4 quirks table.
- **L2 (LOW, acknowledged):** 2016 winner model details listed as "Unknown (logistic variant)" — secondary source not found. Left as-is; adding a note would be speculative.

**Round 1 Outcome:** All ACs implemented. Story status set to `review` — awaiting PO decision gate (AC 7) and SM downstream update (AC 8).

### Senior Developer Review (AI) — Round 2

**Reviewer:** Claude Sonnet 4.6 (adversarial code review — second pass)
**Date:** 2026-02-22
**Issues Found:** 3 HIGH, 3 MEDIUM, 2 LOW
**Issues Fixed:** 6 (all HIGH and MEDIUM; LOW issues L1 also fixed for clarity)
**Git vs Story Discrepancies:** 0

**Summary of fixes applied:**
- **H1 (HIGH):** Added `from ncaa_eval.ingest.schema import Game` to Section 5.2 ABC pseudocode — `Game` was used in `StatefulModel.update(self, game: Game)` without being imported. Round 1 added `Path` but missed `Game`. Same class of copy-paste trap.
- **H2 (HIGH):** Added import block to Section 5.3 evaluation dispatch pseudocode — `StatefulFeatureServer` was used as a type annotation without `from ncaa_eval.transform.feature_serving import StatefulFeatureServer`. Story 6 dev agents will copy this pattern.
- **H3 (HIGH):** Corrected sprint-status from `done` → `review`. Round 1 set sprint-status to `done` but AC 7 (PO gate) has not been passed and Task 8 (SM work) is entirely unchecked. Per Task 7.6/8.4 in the story, `review` is correct until PO approves and SM completes AC 8.
- **M1 (MEDIUM):** Added `early_game_threshold: int = 20` to `EloModelConfig` pseudocode in Section 5.5 — parameter exists in actual `EloConfig` but was omitted from the Pydantic config spec. Story 5.3 dev agents wrapping `EloFeatureEngine` need this to achieve parameter parity.
- **M2 (MEDIUM):** Corrected `XGBClassifier.load_model()` API reference in Section 6.2 — `load_model` is an instance method (not a class method). Corrected to `clf = XGBClassifier(); clf.load_model("model.ubj")`.
- **M3 (MEDIUM):** Added `Callable` return type annotation to `register_model` decorator in Section 5.4 — missing type would cause `mypy --strict` `[no-untyped-def]` failure in Story 5.2.
- **L1 (LOW):** Added `path` parameter to Section 6.1 Elo `save()`/`load()` implementation notes — inconsistency with ABC signature would confuse Story 5.3 dev agent.
- **L2 (LOW):** Removed misleading `# type: ignore[override]` from `StatelessModel.predict()` — mypy does not flag this as an override error; the comment was incorrect and would confuse Story 5.2 dev agents.

**Round 2 Outcome:** Story set to `review`. All ACs remain implemented. Pseudocode is now import-complete and API-accurate across both rounds of review. Awaiting PO decision gate (AC 7) and SM downstream update (AC 8).

### Change Log

- 2026-02-22: Created `specs/research/modeling-approaches.md` with complete survey of modeling approaches for NCAA tournament prediction. 8 sections covering: MMLM solution survey (2014–2025), stateful model catalogue (Elo, Glicko-2, TrueSkill), stateless model catalogue (XGBoost, LightGBM, LR, neural nets), hybrid approaches, Model ABC interface requirements, reference model recommendations, equivalence groups, and scope recommendation for PO decision.
- 2026-02-22: Code review round 1 (Claude Sonnet 4.6) — applied 7 fixes to `specs/research/modeling-approaches.md`: pathlib import in pseudocode, StatelessModel.predict() resolution, arXiv verification caveat, min_child_weight added to XGBoostModelConfig, evaluation pipeline dispatch guidance, stacking code block language tag, https:// URL prefixes, 538 discontinuation entry in Section 1.4. Added Task 8 (post-PO SM work) to story file.
- 2026-02-22: Code review round 2 (Claude Sonnet 4.6) — applied 6 fixes to `specs/research/modeling-approaches.md`: Game import in Section 5.2 ABC pseudocode, StatefulFeatureServer import in Section 5.3 dispatch pseudocode, early_game_threshold added to EloModelConfig (Section 5.5), XGBClassifier.load_model() API correction (Section 6.2), Callable return type on register_model (Section 5.4), save(path)/load(path) consistency in Section 6.1, removed misleading type: ignore[override] comment. Corrected sprint-status from `done` → `review` (PO gate not yet passed).
