# PO Decision Log — Epic 8 Audit Items

Date: 2026-03-05

## Summary

This document records the Product Owner's disposition for all Category 1 (PO direction required) and Category 2 (human judgment required) items identified in the three-pass codebase audit:
- `codebase-audit-report.md` (Pass 1)
- `codebase-audit-pass2-addendum.md` (Pass 2)
- `codebase-audit-pass3-addendum.md` (Pass 3)

**Decision Counts:**
- Category 1: 7 Defer, 6 Accept-as-is, 3 Implement
- Category 2: 10 Accept-as-is, 9 Defer, 3 Already Resolved, 2 Fix

---

## Category 1 Decisions

| # | Item | Decision | Rationale | Follow-up |
|---|------|----------|-----------|-----------|
| 1.1 | Game Theory Sliders | **Defer** | Spike research is done (7.7) but implementation is medium effort. Not critical for core bracket evaluation use case. Already in Post-MVP Backlog #11. | N/A — remains in Post-MVP Backlog |
| 1.2 | No User-Editable Bracket | **Defer** | Medium effort; current read-only bracket + Pool Scorer covers primary use case (evaluate model's bracket). Editing is a nice-to-have. Already in Post-MVP Backlog #12. | N/A — remains in Post-MVP Backlog |
| 1.3 | No Kaggle Submission Export | **Implement** | Low effort (~50 lines), directly supports the project's Kaggle March Mania mission. High value/effort ratio. | Create story in Epic 10 |
| 1.4 | No Model Ensemble/Blending | **Defer** | High effort; requires training multiple independent models first. Single-model XGBoost is competitive. Already in Post-MVP Backlog #17. | N/A — remains in Post-MVP Backlog |
| 1.5 | No Demo/Sample Data | **Defer** | Low effort but the project's primary user is the developer (personal project). Kaggle setup is a one-time cost. Already in Post-MVP Backlog #21. | N/A — remains in Post-MVP Backlog |
| 1.6 | Feature Config Not Configurable from CLI | **Implement** | Medium-low effort; users currently must edit source code to experiment with features. `--feature-config` flag or YAML config path would be a significant usability improvement. | Create story in Epic 10 |
| 1.7 | `team_a_won = True` Label Bias | **Accept-as-is** | The pipeline already warns on label imbalance (>0.95 mean). XGBoost and logistic regression handle this via calibrated probability outputs and are invariant to label permutation when features are symmetric (team_a_X - team_b_X). The current approach is technically sound — the "bias" is cosmetic since predictions are made on symmetric feature differences. | N/A |
| 1.8 | Fibonacci Scoring Values Mismatch | **Accept-as-is** | The code's values (2-3-5-8-13-21) are a better scoring progression for bracket pools than classic Fibonacci (1-1-2-3-5-8) because they avoid the trivial 1-point rounds. The UI label should clarify the actual values (this is the Cat 3 portion from the Pass 2 reclassification, which should be handled as a Cat 3 fix). Accept the code's values as canonical. | Cat 3 fix: update UI label to show actual values |
| 1.9 | Metric Explorer Missing Drill-Downs | **Defer** | Year-only drill-down covers the primary use case (comparing model performance across seasons). Round/seed/conference drill-downs are nice-to-have. Already in Post-MVP Backlog #14. | N/A — remains in Post-MVP Backlog |
| 1.10 | Candidate Entry Flagging | **Defer** | Requires user-editable bracket (1.2) to be meaningful. Already in Post-MVP Backlog #15. | N/A — remains in Post-MVP Backlog |
| 1.11 | CLI `predict` Command | **Defer** | Medium effort; predictions are accessible via dashboard and notebooks. CLI predict is a convenience. Already in Post-MVP Backlog #16. | N/A — remains in Post-MVP Backlog |
| 1.12 | No Per-Game Explainability | **Defer** | High effort (SHAP/LIME). Model-level feature importance (already implemented) covers primary use case. Already in Post-MVP Backlog #20. | N/A — remains in Post-MVP Backlog |
| 1.13 | StatefulModel.fit() Interface Mismatch | **Accept-as-is** | The current approach works correctly even if architecturally impure. The `isinstance` check in backtest is a pragmatic solution. Refactoring would touch core model ABC contract and risk regressions across all model implementations. | N/A |
| 1.14 | Pool Scorer CSV Only | **Defer** | Low effort but CSV covers the primary use case. JSON is nice-to-have for programmatic consumers. Already in Post-MVP Backlog #18. | N/A — remains in Post-MVP Backlog |
| 1.15 | Feature Importance Only XGBoost | **Implement** | Low effort: Elo ratings are inherently interpretable (display team rating values), and LR has `.coef_`. Exposing these as "feature importance" for 2/3 model types significantly improves user understanding. | Create story in Epic 10 |
| P3-17 | NFR3 Plugin Registry 2/4 | **Accept-as-is** | Model and scoring registries cover the extensibility points users actually need. Metric registry is unnecessary — users can compute custom metrics via standard sklearn/numpy. Feature generator registry is high complexity (leakage prevention). The tutorial claim "How to Add a Custom Metric" should be corrected (Cat 3 fix). | Cat 3 fix: correct tutorial claim |

---

## Category 2 Decisions

| # | Item | Decision | Rationale | Follow-up |
|---|------|----------|-----------|-----------|
| 2.1 | `sync.py` at Project Root | **Accept-as-is** | Convenience of `python sync.py` outweighs architectural purity for a personal project. Both entry points work; removing it would break documented examples. | N/A |
| 2.2 | `serving.py` Imports from Ingest | **Accept-as-is** | `ChronologicalDataServer` needs `Repository` and `Game` — this is practical data access, not a layer violation. The import provides type safety for the serving layer's primary data source. | N/A |
| 2.3 | Repository `get_games` Per-Row Construction | **Accept-as-is** | Domain integrity (returning `Game` objects) is more valuable than the minor performance cost. The per-row construction ensures Pydantic validation on every game record. | N/A |
| 2.4 | KaggleConnector Uses `iterrows()` | **Defer** | The ingest layer processes CSV files that are parsed once during sync (not on every request). The `iterrows()` calls in KaggleConnector are not a performance bottleneck — they run during the initial data import, which is a one-time operation per sync. Replacing with vectorized operations is a code quality improvement but not urgent. | Remains in Post-MVP Backlog (via 2.17) |
| 2.5 | Connector ABC Optional Methods | **Accept-as-is** | The "Header Interface" pattern is common in connector ABCs where subclasses support different capabilities. Switching to protocols/mixins would add complexity for 2 concrete implementations. | N/A |
| 2.6 | Giant `__init__.py` Re-exports | **Accept-as-is** | Import convenience outweighs startup time for an interactive dashboard/CLI tool. Users benefit from `from ncaa_eval.transform import EloFeatureEngine` without knowing submodule layout. | N/A |
| 2.7 | EloModelConfig Duplicates EloConfig | **Accept-as-is** | The Pydantic model serves serialization/validation; the frozen dataclass serves runtime immutability. Different purposes justify the duplication. Consolidating would couple model serialization to runtime config. | N/A |
| 2.8 | Model Registry Global Singleton | **Accept-as-is** | Standard pattern for plugin registries (cf. Flask extensions, pytest plugins). Testing isolation is handled by the existing test fixtures. | N/A |
| 2.9 | RunStore Deferred Import | **Accept-as-is** | Deferred imports for circular dependency resolution is a well-established Python pattern. The alternative (restructuring module dependencies) would be a significant refactor for minimal benefit. | N/A |
| 2.10 | Deferred sklearn Imports | **Accept-as-is** | Python caches module imports after the first call. The overhead is ~0.1ms per call after initial import. This is a non-issue. | N/A |
| 2.11 | ESPN Exception Handling | **Already Resolved** | Duplicate of 3.28. Fixed in Story 8.3 with tenacity retry + WARNING-level logging. | N/A |
| 2.12 | `get_data_dir()` `__file__`-Relative Path | **Defer** | Fragile if directory structure moves, but the dashboard directory structure has been stable since Epic 7. Low risk, low priority. | Add to Post-MVP Backlog |
| 2.13 | Dashboard Module-Level `_render_*()` | **Accept-as-is** | This is standard Streamlit convention. All Streamlit apps work this way — code at module level runs on page navigation. Not a bug. | N/A |
| 2.14 | Undocumented Streamlit API | **Defer** | The `event.selection.rows` API is undocumented but widely used in the Streamlit community. Risk is that a Streamlit upgrade breaks it. Low priority — will address if/when Streamlit breaks it. | Add to Post-MVP Backlog |
| 2.15 | Plotly Adapter API Changed from AC | **Accept-as-is** | Standalone functions are a deliberate, documented design decision. The Story explicitly chose functions over methods for composability. Accept the documented deviation. | N/A |
| 2.16 | `st.spinner` Instead of `st.progress` | **Defer** | Already in Post-MVP Backlog #19. `st.spinner` works; `st.progress` is a UX polish. | N/A — remains in Post-MVP Backlog |
| 2.17 | Story 2.3 Open AI-Review Follow-ups | **Defer** | Pandera schema validation and iterrows replacement in KaggleConnector are code quality improvements, not functional bugs. The connector works correctly. | Add to Post-MVP Backlog |
| 2.18 | Top-Level `__init__.py` Missing Re-exports | **Fix** | The Style Guide claims `from ncaa_eval import EloModel` should work but it doesn't. This is a documentation-vs-implementation gap that should be fixed in one direction: either add the re-exports or update the Style Guide. Recommendation: update the Style Guide to document the actual import paths rather than adding re-exports (which would trigger heavy module loading). | Create story in Epic 10 |
| 2.19 | User Guide Documents Sliders As If They Exist | **Already Resolved** | Reclassified to Cat 3 in Pass 2. Addressed in Story 8.4 with "NOT YET IMPLEMENTED" banner. | N/A |
| 2.20 | No Data Post-Sync Validation | **Fix** | Post-sync validation (game count reasonableness, duplicate detection, team reference integrity) would catch silent data corruption. This is a data integrity improvement worth implementing. | Create story in Epic 10 |
| 2.21 | `_make_season_df` Duplicated in Tests | **Defer** | Minor code quality issue — two test files share a small helper. Can be consolidated into a shared fixture when either file is next modified. | Add to Post-MVP Backlog |
| P2-5 | No Coverage Threshold | **Defer** | Need to measure current coverage level before setting a threshold. Setting an arbitrary threshold risks either being too low (useless) or too high (blocks legitimate PRs). Defer until a coverage audit is done. | Add to Post-MVP Backlog |
| P2-6 | Dashboard Excluded from Quality Gates | **Defer** | Streamlit has poor type stubs; strict mypy is impractical. A relaxed mypy config for dashboard/ could catch basics but is a low-priority improvement. | Add to Post-MVP Backlog |
| P3-20 | Architecture Spec Stale | **Accept-as-is** | Story 8.12 already added a historical-document banner. The spec served its purpose during initial design. Updating it to match implementation would be busywork — the code IS the spec now. | N/A |

---

## Follow-up Actions Summary

### Items to Implement (New Stories in Epic 10)

1. **1.3 Kaggle Submission Export** — Low effort, high value. Export bracket to Kaggle MMLM submission format.
2. **1.6 Feature Config from CLI** — Add `--feature-config` CLI option for training pipeline.
3. **1.15 Feature Importance for All Models** — Expose Elo ratings and LR coefficients as feature importance.
4. **2.18 Fix `__init__.py` Public API Documentation** — Update Style Guide to document actual import paths (not re-exports).
5. **2.20 Post-Sync Data Validation** — Add validation checks after data sync.

### Items to Add to Post-MVP Backlog

- **2.12** `get_data_dir()` path fragility
- **2.14** Undocumented Streamlit API usage
- **2.17** Story 2.3 open AI-review follow-ups (Pandera + iterrows)
- **2.21** `_make_season_df` test helper duplication
- **P2-5** Coverage threshold enforcement
- **P2-6** Dashboard quality gate inclusion

### Items Confirmed as Already in Post-MVP Backlog (No Change)

1.1, 1.2, 1.4, 1.5, 1.9, 1.10, 1.11, 1.12, 1.14, 2.4 (via 2.17), 2.16

### Already Resolved Items

- **2.11** — Fixed in Story 8.3 (ESPN exception handling)
- **2.19** — Fixed in Story 8.4 (game theory slider docs)
