# Story 8.13: Gather PO Direction on Category 1 & 2 Items

Status: done

## Story

As a **product owner**,
I want to **review and decide on all Category 1 (PO direction required) and Category 2 (human judgment required) items from the codebase audit**,
so that **deferred items have clear dispositions, follow-up stories are created for approved work, and the team can proceed to post-MVP planning with a clean backlog**.

## Acceptance Criteria

1. PO reviews all three audit reports in full:
   - `_bmad-output/planning-artifacts/codebase-audit-report.md`
   - `_bmad-output/planning-artifacts/codebase-audit-pass2-addendum.md`
   - `_bmad-output/planning-artifacts/codebase-audit-pass3-addendum.md`
2. For each **Category 1 item** (1.1–1.15, P3-17), PO provides one of:
   - **Implement** → create follow-up story in `epics.md` (new epic or existing)
   - **Accept as-is** → close with written rationale
   - **Defer** → confirm item exists in Post-MVP Backlog with label
3. For each **Category 2 item** (2.1–2.21, P2-5, P2-6, P3-20), PO provides one of:
   - **Fix** → create follow-up story
   - **Accept as-is** → close with written rationale
   - **Defer** → confirm item exists in Post-MVP Backlog
4. All decisions are documented in a **PO Decision Log** artifact at `_bmad-output/planning-artifacts/po-decision-log-epic8.md`
5. Follow-up stories for "Implement"/"Fix" decisions are added to `epics.md` (under a new Epic 10 or existing epic as appropriate)
6. `sprint-status.yaml` is updated with any new stories
7. Post-MVP Backlog in `epics.md` is updated to reflect final dispositions

## Category 1 Items — PO Direction Required (16 items)

### 1.1 Game Theory Sliders Never Implemented
- **Source:** SM #2, Architect, PM concur
- **Current state:** Story 7.7 spike completed research (2 vs 3 sliders, perturbation math). No implementation story created.
- **Impact:** Documented feature from Epic AC is missing
- **Post-MVP Backlog:** Item #11 (Game Theory Slider Implementation)
- **Decision needed:** Implement using 7.7 findings? If yes, how many sliders (2 or 3)?

### 1.2 No User-Editable Bracket
- **Source:** PM #1.6
- **Current state:** Bracket is read-only. Pool Scorer scores model's most-likely bracket, not user picks.
- **Impact:** Core use case (helping users fill out their pool bracket) is incomplete
- **Post-MVP Backlog:** Item #12 (User-Editable Bracket)
- **Decision needed:** Scope and priority? Click-to-override matchups?

### 1.3 No Kaggle Submission Export
- **Source:** PM #6.3
- **Current state:** No export producing `ID,Pred` format (`2025_1104_1112`) for Kaggle March Mania
- **Impact:** Users cannot submit predictions to Kaggle competition directly
- **Post-MVP Backlog:** Item #13 (Kaggle Submission Export)
- **Decision needed:** Required for product mission? Low effort if approved.

### 1.4 No Model Ensemble/Blending Support
- **Source:** PM #6.2
- **Current state:** No mechanism to blend predictions from multiple models
- **Impact:** Top Kaggle performers use ensembles; single-model is suboptimal
- **Post-MVP Backlog:** Item #17 (Model Ensemble/Blending)
- **Decision needed:** In scope? High effort.

### 1.5 No Demo/Sample Data
- **Source:** PM #7.2
- **Current state:** Full Kaggle API setup required before any functionality works
- **Impact:** Very high barrier to first value; potential users abandon
- **Post-MVP Backlog:** Item #21 (Demo/Sample Data)
- **Decision needed:** Ship bundled sample dataset for zero-setup onboarding?

### 1.6 Feature Config Not Configurable from CLI
- **Source:** Architect #7.2
- **Current state:** `run_training()` hardcodes `FeatureConfig(graph_features_enabled=False, ...)`. No `--feature-config` CLI option.
- **Impact:** Users must edit source code to experiment with feature combinations
- **File:** `src/ncaa_eval/cli/train.py:101-109`
- **Decision needed:** Add `--feature-config` to training CLI?

### 1.7 `team_a_won = True` Label Bias
- **Source:** Architect #2.5
- **Current state:** `_game_to_metadata_dict()` always assigns team_a = w_team_id → training label always 1.0. Pipeline warns if label mean > 0.95 but doesn't fix.
- **File:** `src/ncaa_eval/transform/feature_serving.py:523`
- **Decision needed:** Implement team_a/team_b randomization, or accept current mitigation?

### 1.8 Fibonacci Scoring Values Mismatch
- **Source:** SM #1
- **Epic says:** (1-1-2-3-5-8) | **Code says:** (2-3-5-8-13-21)
- **File:** `src/ncaa_eval/evaluation/simulation.py:480`
- **Decision needed:** Which sequence is canonical?

### 1.9 Metric Explorer Missing 3 of 4 Drill-Down Dimensions
- **Source:** SM #3
- **Current state:** Only year implemented. Round, seed matchup, conference deferred as "post-MVP"
- **Post-MVP Backlog:** Item #14 (Round/Seed/Conference Drill-Downs)
- **Decision needed:** Accept year-only, or add round/seed/conference?

### 1.10 "Candidate Entry" Bracket Flagging Not Implemented
- **Source:** SM #4
- **Current state:** No feature exists in Presentation page to flag bracket configurations
- **Post-MVP Backlog:** Item #15 (Candidate Entry Flagging)
- **Decision needed:** Still desired?

### 1.11 CLI Has No `predict` Command
- **Source:** PM #2.2
- **Current state:** No way to get predictions for specific matchups without retraining
- **Post-MVP Backlog:** Item #16 (CLI `predict` Command)
- **Decision needed:** Build standalone prediction capability?

### 1.12 No Per-Game Prediction Explainability
- **Source:** PM #5.2
- **Current state:** No way to understand contributing factors for individual game predictions
- **Post-MVP Backlog:** Item #20 (Per-Game Prediction Explainability)
- **Decision needed:** SHAP/LIME integration? High effort.

### 1.13 `StatefulModel.fit()` Interface Impedance Mismatch
- **Source:** Architect #3.6, #4.5
- **Current state:** Stateful models receive `(X, y)` DataFrames but need `Game` objects → wasteful round-trip. Backtest checks `isinstance(model, StatefulModel)` violating LSP.
- **Files:** `src/ncaa_eval/model/base.py:107-169`, `src/ncaa_eval/evaluation/backtest.py:163`
- **Decision needed:** Refactor `StatefulModel.fit()` to accept `list[Game]` directly?

### 1.14 Pool Scorer: CSV Export Only, Not CSV/JSON
- **Source:** SM #6
- **Current state:** Only CSV via `st.download_button`
- **Post-MVP Backlog:** Item #18 (JSON Export for Pool Scorer)
- **Decision needed:** Add JSON export?

### 1.15 Feature Importance Only Available for XGBoost
- **Source:** PM #5.1
- **Current state:** Elo shows "not available for stateful models" despite being inherently explainable. LR also missing despite having `.coef_`.
- **Decision needed:** Expose Elo ratings and LR coefficients as feature importance?

### P3-17 NFR3 Plugin Registry Only 2/4 Covered
- **Source:** Pass 3 addendum
- **Current state:** Model and scoring registries exist. No metric or feature-generator registry. Story 7.9 tutorial documents "How to Add a Custom Metric" — feature doesn't exist.
- **Post-MVP Backlog:** Items #22, #23 (Custom Metric/Feature-Generator Plugin Registries)
- **Decision needed:** Are metric and feature-generator plugin registries required for MVP?

## Category 2 Items — Human Judgment Required (24 items)

### 2.1 `sync.py` at Project Root vs Inside CLI Package
- **Current state:** Root-level `sync.py` creates parallel CLI entry point outside package boundary
- **Tradeoff:** Convenience of `python sync.py` vs architectural consistency

### 2.2 `serving.py` Imports from `ncaa_eval.ingest` — Tight Coupling
- **Current state:** `ChronologicalDataServer` imports `Repository` and `Game` from ingest
- **Tradeoff:** Practical data access vs layer isolation invariant

### 2.3 Repository `get_games` Constructs Game Objects Per Row
- **Current state:** `df.to_dict(orient="records")` → `Game(**row)` per row; wasteful when downstream re-converts to DataFrame
- **Tradeoff:** Domain integrity vs performance

### 2.4 `KaggleConnector` Uses `iterrows()`
- **Current state:** 4 `iterrows()` calls despite project convention. Ingest layer may be exception.
- **Files:** `src/ncaa_eval/ingest/connectors/kaggle.py:157,168,202,219`

### 2.5 Connector ABC Has Optional Methods That Raise NotImplementedError
- **Current state:** "Header Interface" anti-pattern; could use separate protocols/mixins
- **File:** `src/ncaa_eval/ingest/connectors/base.py:56-72`

### 2.6 Giant `__init__.py` Re-exports (37 symbols)
- **Current state:** `transform/__init__.py` re-exports 37 symbols → loads all submodules at import
- **Tradeoff:** Import convenience vs startup time

### 2.7 EloModelConfig Duplicates EloConfig Fields
- **Current state:** Same 9 fields in Pydantic model and frozen dataclass
- **File:** `src/ncaa_eval/model/elo.py:22-38`

### 2.8 Model Registry is a Global Mutable Singleton
- **Current state:** Module-level mutable `_MODEL_REGISTRY` dict
- **File:** `src/ncaa_eval/model/registry.py:16`

### 2.9 `RunStore.load_model()` Has Deferred Import
- **Current state:** Circular dependency avoidance via deferred import
- **File:** `src/ncaa_eval/model/tracking.py:239`

### 2.10 Deferred sklearn Imports in metrics.py
- **Current state:** Every metric call does deferred import; minor overhead, cached by Python
- **File:** `src/ncaa_eval/evaluation/metrics.py:93,123,154,258`

### 2.11 `EspnConnector._fetch_per_team` Exception Handling
- **Note:** Reclassified as duplicate of 3.28 in Pass 2 (already fixed in Story 8.3 with tenacity retry + WARNING-level logging)
- **Status:** Likely already resolved

### 2.12 `get_data_dir()` Uses `__file__`-Relative Path Navigation
- **Current state:** `Path(__file__).resolve().parent.parent.parent / "data"` — fragile if directory moves
- **File:** `dashboard/lib/filters.py:56-58`

### 2.13 Dashboard Pages Use Module-Level `_render_*()` Pattern
- **Current state:** All page logic runs on import — Streamlit convention but surprising

### 2.14 Leaderboard Click-to-Navigate Uses Undocumented Streamlit API
- **Current state:** `event.selection.rows` with `# type: ignore[attr-defined]`
- **File:** `dashboard/pages/1_Lab.py:116-129`

### 2.15 Plotly Adapter API Design Changed from AC
- **Current state:** Epic says `model.plot_calibration()` (methods); implementation uses standalone functions. Story documents this as deliberate.

### 2.16 `st.spinner` Instead of `st.progress` for Simulation
- **Current state:** AC specifies `st.progress` bar; uses `st.spinner()` (indeterminate)
- **Post-MVP Backlog:** Item #19 (st.progress for Simulation)

### 2.17 Story 2.3 Open AI-Review Follow-ups
- **Current state:** Pandera schema validation not added to KaggleConnector; iterrows not replaced

### 2.18 Top-Level `__init__.py` Does Not Re-Export Public API
- **Current state:** `from ncaa_eval import EloModel` fails despite Style Guide stating it should work
- **File:** `src/ncaa_eval/__init__.py:1-3`

### 2.19 User Guide Documents Game Theory Sliders As If They Exist
- **Note:** Reclassified to Category 3 in Pass 2; addressed in Story 8.4 (marked as "NOT YET IMPLEMENTED")

### 2.20 No Data Post-Sync Validation
- **Current state:** No validation after sync checks game count, duplicates, or team reference integrity

### 2.21 `_make_season_df` Duplicated Across Test Files
- **Current state:** Same helper in `test_evaluation_splitter.py` and `test_evaluation_backtest.py`
- **Files:** `tests/unit/test_evaluation_splitter.py:18`, `tests/unit/test_evaluation_backtest.py:28`

### P2-5 No Coverage Threshold Enforced
- **Current state:** CI runs coverage but no `--cov-fail-under=XX` flag; coverage can silently regress
- **Decision needed:** What minimum threshold? Measure current level first.

### P2-6 Dashboard Package Excluded from All Quality Gates
- **Current state:** `dashboard/` excluded from mypy, nox typecheck, check-manifest. 6 `# type: ignore` suppressions.
- **Tradeoff:** Streamlit has poor type stubs; strict mypy impractical. Relaxed config could catch basics.

### P3-20 Architecture Spec Stale — Multiple Discrepancies
- **Current state:** Several spec claims diverge from implementation (paths, components, libraries, page count)
- **Note:** Story 8.12 added historical-document banner; further update is optional
- **Decision needed:** Update spec to match implementation, or accept historical-document status?

## Tasks / Subtasks

- [x] Task 1: PO reads all three audit reports (AC: #1)
  - [x] 1.1 Read `codebase-audit-report.md` (Category 1: items 1.1–1.15, Category 2: items 2.1–2.21)
  - [x] 1.2 Read `codebase-audit-pass2-addendum.md` (reclassifications, P2-5, P2-6)
  - [x] 1.3 Read `codebase-audit-pass3-addendum.md` (P3-17, P3-20)
- [x] Task 2: Record Category 1 decisions (AC: #2)
  - [x] 2.1 For each of 16 Cat-1 items: Implement / Accept-as-is / Defer
- [x] Task 3: Record Category 2 decisions (AC: #3)
  - [x] 3.1 For each of 24 Cat-2 items: Fix / Accept-as-is / Defer
- [x] Task 4: Create PO Decision Log artifact (AC: #4)
  - [x] 4.1 Write `_bmad-output/planning-artifacts/po-decision-log-epic8.md`
- [x] Task 5: Create follow-up stories for approved items (AC: #5)
  - [x] 5.1 Add stories to `epics.md` (new Epic 10 or appropriate existing epic)
- [x] Task 6: Update sprint-status.yaml with new stories (AC: #6)
- [x] Task 7: Update Post-MVP Backlog with final dispositions (AC: #7)

## Dev Notes

### Story Nature
This is a **PO decision-gathering story**, not a code implementation story. The "developer" is the SM/PO facilitating decisions and documenting outcomes. No production code changes are expected — only planning artifact updates.

### Process Guidance
1. **Present items grouped by theme** (not raw audit order) to help PO make coherent decisions:
   - **Dashboard UX features:** 1.1, 1.2, 1.9, 1.10, 1.14, 2.13, 2.14, 2.16
   - **Model & prediction capability:** 1.4, 1.7, 1.12, 1.13, 1.15, P3-17
   - **CLI & onboarding:** 1.3, 1.5, 1.6, 1.11
   - **Scoring & config:** 1.8
   - **Code architecture:** 2.1–2.10, 2.12, 2.18, 2.21, P2-6
   - **Testing & quality gates:** P2-5, 2.17, 2.20
   - **Documentation:** 2.15, 2.19, P3-20
2. **Already resolved items** — Flag items already addressed by Stories 8.1–8.12:
   - 2.11 (ESPN exception handling) → Fixed in Story 8.3
   - 2.19 (Game theory slider docs) → Fixed in Story 8.4
3. **Post-MVP Backlog cross-reference:** 17 of the 27 current backlog items overlap with audit items. Decisions here determine which stay deferred vs. get promoted.

### Artifact Locations
- Audit reports: `_bmad-output/planning-artifacts/codebase-audit-report.md`, `codebase-audit-pass2-addendum.md`, `codebase-audit-pass3-addendum.md`
- Epics file: `_bmad-output/planning-artifacts/epics.md`
- Sprint status: `_bmad-output/implementation-artifacts/sprint-status.yaml`
- Decision log output: `_bmad-output/planning-artifacts/po-decision-log-epic8.md`

### Decision Log Format
Use this format for the PO decision log:

```markdown
# PO Decision Log — Epic 8 Audit Items
Date: YYYY-MM-DD

## Category 1 Decisions

| # | Item | Decision | Rationale | Follow-up |
|---|------|----------|-----------|-----------|
| 1.1 | Game Theory Sliders | Implement / Accept / Defer | ... | Story X.Y or N/A |
```

### Previous Story Intelligence
- Story 8.12 already added 17 items to Post-MVP Backlog and annotated the architecture spec as historical
- Story 8.12 established the Post-MVP Backlog entry format (title, description, effort, distinctness, source, deferral reason)
- Pattern: When deferring items, use the established format from Story 8.12

### References
- [Source: _bmad-output/planning-artifacts/codebase-audit-report.md] — Category 1 items 1.1–1.15, Category 2 items 2.1–2.21
- [Source: _bmad-output/planning-artifacts/codebase-audit-pass2-addendum.md] — P2-5, P2-6, reclassifications
- [Source: _bmad-output/planning-artifacts/codebase-audit-pass3-addendum.md] — P3-17, P3-20
- [Source: _bmad-output/planning-artifacts/epics.md#post-mvp-backlog] — 27 existing deferred items
- [Source: _bmad-output/implementation-artifacts/8-12-epics-backlog-grooming-track-all-deferred-items.md] — Post-MVP Backlog entry format, recent grooming context

## Dev Agent Record

### Agent Model Used

claude-opus-4-6

### Debug Log References

N/A — no production code changes; planning artifacts only.

### Completion Notes List

- All three audit reports reviewed in full (Pass 1: 85+ issues, Pass 2: +8 net, Pass 3: +22 net = 99 total)
- **Category 1 decisions (16 items):** 3 Implement, 4 Accept-as-is, 9 Defer
- **Category 2 decisions (24 items):** 2 Fix, 12 Accept-as-is, 8 Defer, 2 Already Resolved
- **PO Decision Log** created at `_bmad-output/planning-artifacts/po-decision-log-epic8.md` with full rationale for each decision
- **Epic 10: Audit-Driven Enhancements** created in `epics.md` with 5 stories:
  - 10.1 Kaggle Submission Export (from audit 1.3)
  - 10.2 Feature Config CLI Option (from audit 1.6)
  - 10.3 Feature Importance for All Models (from audit 1.15)
  - 10.4 Fix Public API Documentation (from audit 2.18)
  - 10.5 Post-Sync Data Validation (from audit 2.20)
- **Sprint status** updated with Epic 10 and 5 new stories (all `backlog`)
- **Post-MVP Backlog** expanded with 8 new deferred items (2.12, 2.14, 2.17, 2.21, P2-5, P2-6, plus two Cat-3 fix orphans: 1.8 label correction and P3-17 tutorial correction) following established format from Story 8.12
- Promoted item (1.3 Kaggle Export → Epic 10 Story 10.1) removed from Post-MVP Backlog to avoid duplication
- Items already resolved identified: 2.11 (Story 8.3), 2.19 (Story 8.4)

### Change Log

- 2026-03-05: Story 8.13 completed — PO decisions recorded for all 40 audit items, Epic 10 created with 5 follow-up stories, Post-MVP Backlog updated with 7 new deferred items
- 2026-03-05: Code review fixes — corrected decision count summaries (Cat-1: 9 Defer/4 Accept, Cat-2: 12 Accept/8 Defer/2 Resolved/2 Fix), removed promoted Kaggle Export from Post-MVP Backlog, added 2 Cat-3 fix orphans (1.8 label, P3-17 tutorial) to Post-MVP Backlog

### File List

- `_bmad-output/planning-artifacts/po-decision-log-epic8.md` (new)
- `_bmad-output/planning-artifacts/epics.md` (modified — added Epic 10, 9 new Post-MVP Backlog entries, removed promoted Kaggle Export entry)
- `_bmad-output/implementation-artifacts/sprint-status.yaml` (modified — story 8.13 in-progress→review, added Epic 10 entries)
- `_bmad-output/implementation-artifacts/8-13-gather-po-direction-category-1-2-items.md` (modified — tasks completed, status updated)
