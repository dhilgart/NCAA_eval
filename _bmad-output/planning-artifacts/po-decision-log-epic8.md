# PO Decision Log — Epic 8 Audit Items

Date: 2026-03-05

## Summary

This document records the Product Owner's disposition for all Category 1 (PO direction required) and Category 2 (human judgment required) items identified in the three-pass codebase audit:
- `codebase-audit-report.md` (Pass 1)
- `codebase-audit-pass2-addendum.md` (Pass 2)
- `codebase-audit-pass3-addendum.md` (Pass 3)

**Decision Counts:**
- Category 1: 9 Defer, 4 Accept-as-is, 3 Implement
- Category 2: 12 Accept-as-is, 8 Defer, 2 Already Resolved, 2 Fix

---

## Category 1 Decisions

---

### 1.1 Game Theory Sliders

**Decision needed:** Should the Game Theory sliders (Upset Aggression, Chalk Bias, Seed-Weight) be implemented for the dashboard, given that Epic 7 AC specified them but Story 7.5 deferred them and the 7.7 spike only completed research?

**Context:** The Epic 7 AC states: "Game Theory sliders in the sidebar perturb the model's base probabilities in real-time." Story 7.5 marked them OUT OF SCOPE and Story 7.7 completed a spike research. No follow-up implementation story was ever created. The feature is documented in the user guide (with a "NOT YET IMPLEMENTED" banner added in Story 8.4). This is already tracked as Post-MVP Backlog #11.

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **A. Implement now (Epic 10)** | Fulfills original Epic 7 AC; spike research already done; adds differentiated UX | Medium effort; not critical for core bracket evaluation; delays other higher-value work |
| **B. Defer to post-MVP** | Keeps focus on core evaluation use case; spike is preserved for future use | AC remains unfulfilled; documented feature still missing |
| **C. Drop entirely** | Removes maintenance burden of stale docs/backlog items | Loses a differentiated feature; wastes spike research investment |

**Recommendation:** B. Defer to post-MVP

**Rationale:** Spike research is done (7.7) but implementation is medium effort. Not critical for the core bracket evaluation use case. The spike investment is preserved and can be picked up post-MVP. Already tracked in Post-MVP Backlog #11.

**Follow-up:** N/A — remains in Post-MVP Backlog

---

### 1.2 No User-Editable Bracket

**Decision needed:** Should users be able to click matchups to override model picks in the bracket view, and should the Pool Scorer score user-edited brackets (not just the model's most-likely bracket)?

**Context:** Users cannot click matchups to override model picks. The Pool Scorer only scores the model's most-likely bracket. The PM identified this as a gap in the core use case — helping users fill out their March Madness pool bracket. The Architect notes that implementing this requires a new `UserOverrideProvider` that wraps an existing `ProbabilityProvider` and substitutes user picks at specific bracket nodes — a non-trivial architectural extension, not a simple UI feature. Already tracked as Post-MVP Backlog #12.

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **A. Implement now (Epic 10)** | Completes the core user journey (evaluate → edit → submit bracket); highest product value feature | Medium-high effort; requires new `UserOverrideProvider` architecture; non-trivial state management in Streamlit |
| **B. Defer to post-MVP** | Current read-only bracket + Pool Scorer covers the primary "evaluate model's bracket" use case | Core use case (user filling out their own bracket) remains incomplete |
| **C. Implement a simplified version (lock/unlock picks)** | Lower effort than full edit; still gives users agency over bracket | May feel half-baked; still requires some architectural work |

**Recommendation:** B. Defer to post-MVP

**Rationale:** Medium effort; the current read-only bracket + Pool Scorer covers the primary use case (evaluate the model's bracket). Editing is a nice-to-have. The project's primary user is the developer, who can interpret model probabilities directly. Already tracked in Post-MVP Backlog #12.

**Follow-up:** N/A — remains in Post-MVP Backlog

---

### 1.3 No Kaggle Submission Export

**Decision needed:** Should the project support exporting predictions in Kaggle March Mania submission format (`ID,Pred` with format `2025_1104_1112`)?

**Context:** The product's tagline references Kaggle March Mania, but there is no export producing the required Kaggle `ID,Pred` format. Users cannot submit predictions to the Kaggle competition directly. The existing `export_bracket_csv` in `dashboard/lib/filters.py` exports bracket data but not in Kaggle's required format. This is estimated as low effort (~50 lines).

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **A. Implement now (Epic 10)** | Low effort (~50 lines); directly supports the project's Kaggle March Mania mission; high value/effort ratio | Minor scope addition to Epic 10 |
| **B. Defer to post-MVP** | Keeps Epic 10 scope minimal | Misses the 2026 March Mania competition window; contradicts project tagline |
| **C. Implement as CLI-only (no dashboard)** | Even lower effort; avoids dashboard changes | Users must use CLI for a common operation |

**Recommendation:** A. Implement now (Epic 10)

**Rationale:** Low effort (~50 lines), directly supports the project's Kaggle March Mania mission. High value/effort ratio. The project exists to support Kaggle competition participation, so this is a core feature gap.

**Follow-up:** Create story in Epic 10

---

### 1.4 No Model Ensemble/Blending Support

**Decision needed:** Should the system support blending predictions from multiple models (e.g., 60% XGBoost + 40% Elo)?

**Context:** No mechanism exists to blend predictions from multiple models. Top Kaggle performers universally use ensembles; single-model predictions are strictly suboptimal. However, building ensembles requires first training multiple independent models with different architectures and hyperparameters. The current system supports three model types (XGBoost, Logistic Regression, Elo) but no composition layer.

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **A. Implement now (Epic 10)** | Directly improves Kaggle leaderboard competitiveness; architecturally interesting | High effort; requires training multiple independent models first; complex UX for model selection/weighting |
| **B. Defer to post-MVP** | Single-model XGBoost is competitive; keeps scope manageable | Missing a well-known competitive advantage |
| **C. Implement simple average only** | Lower effort than weighted blending; still captures ensemble benefit | Less flexible; limited competitive advantage over weighted approach |

**Recommendation:** B. Defer to post-MVP

**Rationale:** High effort; requires training multiple independent models first. Single-model XGBoost is competitive for a personal project. Already tracked in Post-MVP Backlog #17.

**Follow-up:** N/A — remains in Post-MVP Backlog

---

### 1.5 No Demo/Sample Data

**Decision needed:** Should the project include bundled sample data so users can explore the dashboard without a full Kaggle API setup?

**Context:** The product requires full Kaggle API setup (account, phone verification, competition rules acceptance, API token) before any functionality works. This creates a very high barrier to first value. Potential users may abandon before seeing the product. Estimated as low effort.

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **A. Implement now (Epic 10)** | Dramatically lowers onboarding barrier; enables demos and screenshots; low effort | Adds data files to the repo; sample data may go stale across seasons |
| **B. Defer to post-MVP** | Keeps repo lean; primary user (developer) already has Kaggle setup | High barrier to entry persists for any external users |
| **C. Generate synthetic data instead of real data** | Avoids any licensing/data-size concerns; deterministic | Less realistic; harder to validate dashboard behavior |

**Recommendation:** B. Defer to post-MVP

**Rationale:** Low effort but the project's primary user is the developer (personal project). Kaggle setup is a one-time cost. Already tracked in Post-MVP Backlog #21.

**Follow-up:** N/A — remains in Post-MVP Backlog

---

### 1.6 Feature Config Not Configurable from CLI

**Decision needed:** Should users be able to specify feature engineering configuration (e.g., graph features, batch rating types) via the CLI instead of editing source code?

**Context:** `run_training()` in `src/ncaa_eval/cli/train.py:101-109` hardcodes `FeatureConfig(graph_features_enabled=False, batch_rating_types=(), ...)`. No `--feature-config` CLI option exists. Users must edit source code to experiment with different feature combinations, which is a significant usability friction for the primary use case of training/tuning models.

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **A. Implement `--feature-config` flag (YAML path)** | Users point to a YAML file with feature settings; most flexible; supports version-controlled configs | Requires YAML schema definition; slightly higher effort |
| **B. Implement individual CLI flags** | Direct CLI args like `--graph-features`, `--batch-ratings elo,glicko`; discoverable via `--help` | Many flags for all config options; less composable than YAML |
| **C. Defer to post-MVP** | No work needed now | Users continue editing source code to experiment |

**Recommendation:** A. Implement `--feature-config` flag (YAML path)

**Rationale:** Medium-low effort; users currently must edit source code to experiment with features. A `--feature-config` flag or YAML config path would be a significant usability improvement. YAML approach is more maintainable as feature options grow.

**Follow-up:** Create story in Epic 10

---

### 1.7 `team_a_won = True` Label Bias

**Decision needed:** Should the feature server randomize team_a/team_b assignment during training, given that `_game_to_metadata_dict()` always assigns team_a = w_team_id, making the training label always 1.0?

**Context:** In `src/ncaa_eval/transform/feature_serving.py:523`, the function always assigns the winning team as team_a, so the label is always 1.0. The pipeline warns if label mean > 0.95 but doesn't fix it. However, predictions are made on symmetric feature differences (team_a_X - team_b_X), meaning XGBoost and logistic regression outputs are invariant to label permutation when features are constructed this way.

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **A. Randomize team_a/team_b assignment** | Eliminates cosmetic label bias; training data looks more natural; some models may learn more robust decision boundaries | Risk of introducing bugs; adds complexity; current approach is technically sound for symmetric features |
| **B. Accept as-is** | No risk of regressions; technically correct for symmetric feature differences; simpler code | Cosmetic label imbalance warning persists; could confuse users inspecting training data |
| **C. Suppress the warning for this known case** | Removes noise from pipeline output; no behavior change | Hides a potentially useful diagnostic for future model types |

**Recommendation:** B. Accept as-is

**Rationale:** The pipeline already warns on label imbalance (>0.95 mean). XGBoost and logistic regression handle this via calibrated probability outputs and are invariant to label permutation when features are symmetric (team_a_X - team_b_X). The current approach is technically sound — the "bias" is cosmetic since predictions are made on symmetric feature differences.

**Follow-up:** N/A

---

### 1.8 Fibonacci Scoring Values Mismatch

**Decision needed:** Which scoring values are canonical — the Epic AC's classic Fibonacci (1-1-2-3-5-8) or the code's modified sequence (2-3-5-8-13-21)? And should the UI label match the actual values?

**Context:** The Epic AC specifies Fibonacci scoring as (1-1-2-3-5-8). The code in `src/ncaa_eval/evaluation/simulation.py:480` uses (2-3-5-8-13-21). The PM flagged this as a data integrity risk: if a user selects "Fibonacci" scoring expecting classic values and gets different ones, their pool standings calculations will be silently wrong. Pass 2 reclassified the UI label portion as a Category 3 bug (label must match code regardless of which values are chosen).

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **A. Keep code values (2-3-5-8-13-21), fix UI label** | Better scoring progression for bracket pools (avoids trivial 1-point rounds); only requires a UI label fix (Cat 3) | Deviates from classic Fibonacci; "Fibonacci" name is misleading |
| **B. Revert to classic Fibonacci (1-1-2-3-5-8)** | Matches the mathematical definition and Epic AC; no user confusion about what "Fibonacci" means | Trivial 1-point rounds in R64/R32 reduce scoring differentiation |
| **C. Rename to "Progressive" or "Weighted" scoring, keep code values** | Honest labeling; keeps the better scoring progression; removes Fibonacci confusion | Breaks name continuity with Epic AC |

**Recommendation:** A. Keep code values (2-3-5-8-13-21), fix UI label

**Rationale:** The code's values (2-3-5-8-13-21) are a better scoring progression for bracket pools than classic Fibonacci (1-1-2-3-5-8) because they avoid the trivial 1-point rounds. The UI label should clarify the actual values. The Cat 3 portion (update UI label to show actual values) should be handled as a Category 3 fix.

**Follow-up:** Cat 3 fix: update UI label to show actual values (e.g., "Fibonacci (2-3-5-8-13-21)")

---

### 1.9 Metric Explorer Missing Drill-Downs

**Decision needed:** Should the Metric Explorer implement the remaining 3 of 4 drill-down dimensions (round, seed matchup, conference) specified in the Epic AC, or is year-only drill-down sufficient?

**Context:** The Epic AC says: "drill-down by year, round, seed matchup, or conference." Only year is implemented in `dashboard/pages/3_Model_Deep_Dive.py:33`. Round, seed matchup, and conference were explicitly deferred as "post-MVP." Already tracked as Post-MVP Backlog #14.

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **A. Implement all three remaining drill-downs** | Fulfills Epic AC completely; provides rich model analysis capabilities | Medium-high effort; requires data aggregation logic for each dimension; may clutter the UI |
| **B. Implement round drill-down only** | Round is the most analytically valuable dimension (early-round vs. late-round accuracy); moderate effort | Still leaves 2 of 4 AC items unfulfilled |
| **C. Defer all to post-MVP** | Year-only covers the primary use case (comparing model performance across seasons); keeps scope focused | AC remains partially unfulfilled |

**Recommendation:** C. Defer all to post-MVP

**Rationale:** Year-only drill-down covers the primary use case (comparing model performance across seasons). Round/seed/conference drill-downs are nice-to-have. Already tracked in Post-MVP Backlog #14.

**Follow-up:** N/A — remains in Post-MVP Backlog

---

### 1.10 Candidate Entry Flagging

**Decision needed:** Should users be able to flag a specific bracket configuration as a "Candidate Entry" in the Presentation page, as specified in the Epic AC?

**Context:** The Epic AC says: "the user can flag a specific bracket configuration as a 'Candidate Entry'." No such feature exists in the Presentation page. This feature is most meaningful when combined with a user-editable bracket (1.2), since without editing there is only one bracket to "flag." Already tracked as Post-MVP Backlog #15.

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **A. Implement now** | Fulfills Epic AC; enables workflow of saving/comparing bracket states | Requires user-editable bracket (1.2) to be meaningful; low standalone value |
| **B. Defer to post-MVP** | Depends on 1.2 which is also deferred; no standalone value without editable bracket | AC remains unfulfilled |
| **C. Drop entirely** | Simplifies scope; editing + flagging can be redesigned together later | Loses planned feature |

**Recommendation:** B. Defer to post-MVP

**Rationale:** Requires user-editable bracket (1.2) to be meaningful. Without the ability to edit brackets, there is only one configuration to flag. Already tracked in Post-MVP Backlog #15.

**Follow-up:** N/A — remains in Post-MVP Backlog

---

### 1.11 CLI `predict` Command

**Decision needed:** Should the CLI include a `predict` command that generates predictions for specific matchups without retraining?

**Context:** During March Madness, users need predictions for specific matchups NOW without retraining. The PM identified this as a gap: the primary use case (tournament predictions) is not accessible via CLI. However, predictions are accessible via the dashboard and notebooks.

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **A. Implement now (Epic 10)** | Enables scripted/automated prediction workflows; useful during tournament | Medium effort; predictions already accessible via dashboard and notebooks |
| **B. Defer to post-MVP** | Dashboard and notebooks cover the use case for the primary user | CLI-driven workflows not supported |
| **C. Implement as a minimal "predict from saved model" command** | Lower effort; just loads a saved model and runs inference | Still requires model serialization/loading to work correctly |

**Recommendation:** B. Defer to post-MVP

**Rationale:** Medium effort; predictions are accessible via the dashboard and notebooks. CLI predict is a convenience feature. Already tracked in Post-MVP Backlog #16.

**Follow-up:** N/A — remains in Post-MVP Backlog

---

### 1.12 No Per-Game Explainability

**Decision needed:** Should the system provide per-game prediction explanations (e.g., "Duke has a 72% chance of beating UNC because of X, Y, Z factors")?

**Context:** Users see probability predictions but have no way to understand contributing factors for individual games. The PM notes this limits users' ability to make informed decisions about when to trust vs. override the model. SHAP/LIME integration would be the standard approach. Model-level feature importance is already implemented for XGBoost. Already tracked as Post-MVP Backlog #20.

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **A. Implement SHAP explanations** | Gold standard for ML explainability; per-game feature attribution | High effort; SHAP adds dependency and computation time; requires careful integration with all 3 model types |
| **B. Implement simple feature-delta display** | Show raw feature values for both teams side-by-side; low effort | Not true explainability; doesn't show feature contribution to the prediction |
| **C. Defer to post-MVP** | Model-level feature importance (already implemented) covers the primary use case | Users can't understand individual game predictions |

**Recommendation:** C. Defer to post-MVP

**Rationale:** High effort (SHAP/LIME). Model-level feature importance (already implemented) covers the primary use case. Already tracked in Post-MVP Backlog #20.

**Follow-up:** N/A — remains in Post-MVP Backlog

---

### 1.13 StatefulModel.fit() Interface Mismatch

**Decision needed:** Should `StatefulModel.fit()` accept `list[Game]` directly instead of `(X, y)` DataFrames, given that stateful models (Elo) need `Game` objects and the current approach causes wasteful round-trip serialization?

**Context:** Stateful models receive `(X, y)` DataFrames but need `Game` objects, causing wasteful round-trip serialization. The backtest module checks `isinstance(model, StatefulModel)` in `src/ncaa_eval/evaluation/backtest.py:163`, violating LSP. However, the current approach works correctly. Refactoring would touch the core Model ABC contract and risk regressions across all model implementations.

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **A. Refactor to accept `list[Game]` directly** | Cleaner architecture; eliminates wasteful serialization; removes isinstance check | High risk; touches core Model ABC contract; risk of regressions across all model implementations |
| **B. Accept as-is** | Works correctly despite being architecturally impure; no regression risk; `isinstance` check is pragmatic | Maintains technical debt; violates LSP |
| **C. Add a `StatefulModel.fit_games()` method** | Preserves backward compatibility; gives stateful models a clean path; no isinstance needed | Adds API surface; two ways to do the same thing |

**Recommendation:** B. Accept as-is

**Rationale:** The current approach works correctly even if architecturally impure. The `isinstance` check in backtest is a pragmatic solution. Refactoring would touch the core Model ABC contract and risk regressions across all model implementations.

**Follow-up:** N/A

---

### 1.14 Pool Scorer CSV Only

**Decision needed:** Should the Pool Scorer support JSON export in addition to CSV, as specified in the Epic AC ("CSV/JSON")?

**Context:** The Epic AC specifies "CSV/JSON" export. The implementation in `dashboard/pages/4_Pool_Scorer.py` only provides CSV via `st.download_button`. JSON would be useful for programmatic consumers but CSV covers the primary use case (spreadsheet import).

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **A. Implement JSON export** | Fulfills Epic AC; useful for programmatic consumers; low effort | Minor scope addition; few users need programmatic access |
| **B. Defer to post-MVP** | CSV covers the primary use case (spreadsheet import); keeps scope minimal | AC partially unfulfilled |
| **C. Drop JSON requirement** | Simplifies scope permanently | Loses programmatic export capability |

**Recommendation:** B. Defer to post-MVP

**Rationale:** Low effort but CSV covers the primary use case. JSON is nice-to-have for programmatic consumers. Already tracked in Post-MVP Backlog #18.

**Follow-up:** N/A — remains in Post-MVP Backlog

---

### 1.15 Feature Importance Only XGBoost

**Decision needed:** Should feature importance be exposed for Elo and Logistic Regression models, which currently show "Feature importance is not available for stateful models" despite being inherently explainable?

**Context:** The dashboard shows "Feature importance is not available for stateful models" for Elo, despite Elo being inherently explainable (team ratings). Logistic Regression also has `.coef_` for feature importance but it is not exposed. Users can't understand model behavior for 2 of 3 model types. The implementation requires: Elo — display team rating values; LR — expose `.coef_` as feature weights.

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **A. Implement for all model types** | Low effort: Elo ratings are inherently interpretable, LR has `.coef_`; significantly improves user understanding of 2/3 model types | Requires defining what "feature importance" means for each model type (not directly comparable across types) |
| **B. Implement for LR only (has `.coef_`)** | Very low effort; LR coefficients are standard feature importance | Elo still shows "not available" despite being the most interpretable model |
| **C. Defer to post-MVP** | No work needed now | Users can't understand 2 of 3 model types |

**Recommendation:** A. Implement for all model types

**Rationale:** Low effort: Elo ratings are inherently interpretable (display team rating values), and LR has `.coef_`. Exposing these as "feature importance" for 2/3 model types significantly improves user understanding.

**Follow-up:** Create story in Epic 10

---

### P3-17 NFR3 Plugin Registry 2/4

**Decision needed:** Are metric and feature-generator plugin registries required for MVP, given that the PRD (NFR3) specifies 4 registries but only model and scoring registries are implemented?

**Context:** The PRD says NFR3 requires plugin registries for: (1) models, (2) scoring functions, (3) metrics, (4) feature generators. Only model and scoring registries exist. No metric registry or feature-generator registry has been implemented or has any story. Story 7.9 includes a "How to Add a Custom Metric" tutorial — documenting a feature that doesn't exist. The PM considers this a product requirement gap, not just tech debt.

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **A. Implement both metric and feature-generator registries** | Fully satisfies NFR3; makes tutorial accurate | High complexity, especially for feature-generator registry (leakage prevention is hard); metric registry is unnecessary given sklearn/numpy |
| **B. Accept 2/4 coverage as sufficient** | Model and scoring registries cover the extensibility points users actually need; avoids over-engineering | NFR3 not fully satisfied; tutorial claim is inaccurate |
| **C. Implement metric registry only, drop feature-generator** | Moderate effort; makes the tutorial accurate; metric registry is simpler | Feature-generator registry still missing from NFR3 |

**Recommendation:** B. Accept 2/4 coverage as sufficient

**Rationale:** Model and scoring registries cover the extensibility points users actually need. Metric registry is unnecessary — users can compute custom metrics via standard sklearn/numpy. Feature generator registry is high complexity (leakage prevention). The tutorial claim "How to Add a Custom Metric" should be corrected as a Cat 3 fix.

**Follow-up:** Cat 3 fix: correct tutorial claim about custom metrics

---

## Category 2 Decisions

---

### 2.1 `sync.py` at Project Root

**Decision needed:** Should the root-level `sync.py` convenience script be removed in favor of the package-internal CLI entry point (`ncaa-eval sync`)?

**Context:** A root-level `sync.py` creates a parallel CLI entry point outside the package boundary. The official CLI command is `ncaa-eval sync` (via the typer CLI package). The sidebar in `dashboard/app.py` references `python sync.py`. Pass 2 found this inconsistency confuses new users (P2-8). The convenience of `python sync.py` is valued for a personal project.

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **A. Remove `sync.py`, use CLI only** | Single canonical entry point; cleaner architecture; no user confusion | Breaks documented examples; loses convenience of `python sync.py` |
| **B. Accept as-is** | Convenience of `python sync.py` outweighs architectural purity for a personal project; both entry points work | Parallel entry points cause confusion; sidebar references wrong command |
| **C. Keep `sync.py` but update docs to reference CLI** | Both work; docs are consistent; advanced users can use either | Still two entry points |

**Recommendation:** B. Accept as-is

**Rationale:** Convenience of `python sync.py` outweighs architectural purity for a personal project. Both entry points work; removing it would break documented examples.

**Follow-up:** N/A

---

### 2.2 `serving.py` Imports from Ingest

**Decision needed:** Does `ChronologicalDataServer` importing `Repository` and `Game` from the ingest layer constitute a problematic layer violation?

**Context:** `src/ncaa_eval/transform/serving.py:19-20` imports `Repository` and `Game` from `ncaa_eval.ingest`, breaking the "no ingest imports" invariant documented in other transform modules. However, the data server needs these types to access its primary data source.

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **A. Refactor to use an abstract data source interface** | Clean layer separation; transform layer doesn't depend on ingest | Over-engineering for 1 concrete implementation; adds complexity |
| **B. Accept as-is** | Practical data access; provides type safety; `Repository` and `Game` are stable types | Technically violates the documented layer boundary |
| **C. Move `Game` to a shared types module** | Resolves the layer violation for the data model; ingest and transform both import from shared | Refactoring effort; may cause import chain changes |

**Recommendation:** B. Accept as-is

**Rationale:** `ChronologicalDataServer` needs `Repository` and `Game` — this is practical data access, not a layer violation. The import provides type safety for the serving layer's primary data source.

**Follow-up:** N/A

---

### 2.3 Repository `get_games` Per-Row Construction

**Decision needed:** Should `Repository.get_games()` return raw DataFrames instead of constructing `Game` objects per row, given that downstream consumers often convert back to DataFrames?

**Context:** `df.to_dict(orient="records")` → `Game(**row)` for every row. This is wasteful when downstream consumers immediately convert back to DataFrames. However, the per-row construction ensures Pydantic validation on every game record, catching data integrity issues early.

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **A. Return raw DataFrames** | Better performance; avoids round-trip serialization | Loses Pydantic validation; data integrity issues caught later (or never) |
| **B. Accept as-is** | Domain integrity via Pydantic validation on every record; catches bad data early | Minor performance cost for per-row construction |
| **C. Add a `get_games_df()` bypass method** | Callers choose: validated objects or raw performance | Two methods for similar purpose; maintenance burden |

**Recommendation:** B. Accept as-is

**Rationale:** Domain integrity (returning `Game` objects) is more valuable than the minor performance cost. The per-row construction ensures Pydantic validation on every game record.

**Follow-up:** N/A

---

### 2.4 KaggleConnector Uses `iterrows()`

**Decision needed:** Should the `iterrows()` calls in KaggleConnector be replaced with vectorized operations, despite the project's "no iterrows()" convention?

**Context:** `src/ncaa_eval/ingest/connectors/kaggle.py` uses `iterrows()` at lines 157, 168, 202, 219 despite the project's convention against it. However, the ingest layer processes CSV files that are parsed once during sync (not on every request). These calls are not a performance bottleneck — they run during the initial data import, which is a one-time operation per sync.

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **A. Replace with vectorized operations** | Consistent with project convention; better code quality | Not a performance bottleneck; ingest runs once per sync; effort for minimal benefit |
| **B. Defer to post-MVP** | Low priority; code works correctly; focus on higher-value items | Convention violation persists |
| **C. Add explicit exception to convention** | Documents why ingest layer is allowed to use iterrows; avoids future audit findings | Convention becomes less clear-cut |

**Recommendation:** B. Defer to post-MVP

**Rationale:** The `iterrows()` calls in KaggleConnector are not a performance bottleneck — they run during the initial data import, which is a one-time operation per sync. Replacing with vectorized operations is a code quality improvement but not urgent.

**Follow-up:** Remains in Post-MVP Backlog (via 2.17)

---

### 2.5 Connector ABC Optional Methods

**Decision needed:** Should the Connector ABC's optional methods (that raise `NotImplementedError`) be replaced with separate protocols or mixins?

**Context:** `src/ncaa_eval/ingest/connectors/base.py:56-72` uses the "Header Interface" anti-pattern where optional methods raise `NotImplementedError`. This is common in connector ABCs where subclasses support different capabilities. There are only 2 concrete implementations.

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **A. Refactor to protocols/mixins** | Cleaner interface segregation; callers can type-check capabilities | Adds complexity for only 2 concrete implementations; over-engineering |
| **B. Accept as-is** | Simple; common pattern; works for the current scale | "Header Interface" anti-pattern; not type-safe for optional capabilities |

**Recommendation:** B. Accept as-is

**Rationale:** The "Header Interface" pattern is common in connector ABCs where subclasses support different capabilities. Switching to protocols/mixins would add complexity for 2 concrete implementations.

**Follow-up:** N/A

---

### 2.6 Giant `__init__.py` Re-exports

**Decision needed:** Should `transform/__init__.py`'s 37-symbol re-export be reduced to avoid loading all submodules (including heavy dependencies like networkx and sklearn) on import?

**Context:** `transform/__init__.py` re-exports 37 symbols, triggering loading of all submodules (networkx, sklearn, etc.) on any import from the transform package. This affects startup time for the dashboard and CLI.

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **A. Remove re-exports, require explicit submodule imports** | Faster startup; only load what you need | Breaks existing import patterns; less convenient |
| **B. Accept as-is** | Import convenience; users benefit from `from ncaa_eval.transform import EloFeatureEngine` without knowing submodule layout | Slower startup; all submodules loaded even if unused |
| **C. Use lazy imports (`__getattr__`)** | Best of both worlds: convenience + lazy loading | More complex; harder to debug import errors |

**Recommendation:** B. Accept as-is

**Rationale:** Import convenience outweighs startup time for an interactive dashboard/CLI tool. Users benefit from `from ncaa_eval.transform import EloFeatureEngine` without knowing submodule layout.

**Follow-up:** N/A

---

### 2.7 EloModelConfig Duplicates EloConfig

**Decision needed:** Should `EloModelConfig` (Pydantic model) and `EloConfig` (frozen dataclass) be consolidated, given that they share the same 9 fields?

**Context:** In `src/ncaa_eval/model/elo.py:22-38`, both a Pydantic model and a frozen dataclass define the same 9 Elo configuration fields. The Pydantic model serves serialization/validation; the frozen dataclass serves runtime immutability. They serve different purposes.

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **A. Consolidate into a single Pydantic model** | DRY; single source of truth for Elo config | Pydantic models are mutable by default; loses runtime immutability guarantee; couples serialization to runtime |
| **B. Accept as-is** | Different purposes justify the duplication; Pydantic for serialization, dataclass for immutability | 9 duplicated fields; must keep in sync manually |
| **C. Have EloConfig inherit from EloModelConfig** | Reduces duplication while preserving different behaviors | Tight coupling between serialization and runtime types |

**Recommendation:** B. Accept as-is

**Rationale:** The Pydantic model serves serialization/validation; the frozen dataclass serves runtime immutability. Different purposes justify the duplication. Consolidating would couple model serialization to runtime config.

**Follow-up:** N/A

---

### 2.8 Model Registry Global Singleton

**Decision needed:** Should the module-level mutable `_MODEL_REGISTRY` dict be replaced with a dependency-injected or scoped registry?

**Context:** `src/ncaa_eval/model/registry.py:16` uses a module-level mutable `_MODEL_REGISTRY` dict. This is a standard pattern for plugin registries (cf. Flask extensions, pytest plugins) but can make testing harder due to shared global state.

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **A. Refactor to dependency-injected registry** | Better testability; no global state; explicit dependencies | Over-engineering for a plugin registry; adds boilerplate to every call site |
| **B. Accept as-is** | Standard pattern; testing isolation handled by existing test fixtures | Global mutable state; tests must clean up |

**Recommendation:** B. Accept as-is

**Rationale:** Standard pattern for plugin registries (cf. Flask extensions, pytest plugins). Testing isolation is handled by the existing test fixtures.

**Follow-up:** N/A

---

### 2.9 RunStore Deferred Import

**Decision needed:** Should the deferred import in `RunStore.load_model()` be refactored to avoid the circular dependency?

**Context:** `src/ncaa_eval/model/tracking.py:239` uses `from ncaa_eval.model.registry import get_model` inside the method body to avoid a circular import at module level. This is a well-established Python pattern for circular dependency resolution.

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **A. Restructure module dependencies** | Eliminates circular dependency; cleaner imports | Significant refactor for minimal benefit; may introduce new import chains |
| **B. Accept as-is** | Well-established Python pattern; works reliably; minimal overhead | Deferred import is slightly surprising to readers |

**Recommendation:** B. Accept as-is

**Rationale:** Deferred imports for circular dependency resolution is a well-established Python pattern. The alternative (restructuring module dependencies) would be a significant refactor for minimal benefit.

**Follow-up:** N/A

---

### 2.10 Deferred sklearn Imports

**Decision needed:** Should the deferred sklearn imports in `metrics.py` be moved to module level?

**Context:** Every call to `log_loss()`, `brier_score()`, etc. in `src/ncaa_eval/evaluation/metrics.py` does a deferred import. Python caches module imports after the first call, so the overhead is ~0.1ms per call after initial import.

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **A. Move to module-level imports** | Cleaner code; conventional import style | May slow module import for consumers that don't need sklearn |
| **B. Accept as-is** | Negligible overhead after first call; avoids loading sklearn for non-metric use cases | Slightly unconventional import pattern |

**Recommendation:** B. Accept as-is

**Rationale:** Python caches module imports after the first call. The overhead is ~0.1ms per call after initial import. This is a non-issue.

**Follow-up:** N/A

---

### 2.11 ESPN Exception Handling

**Decision needed:** N/A — this item is a duplicate.

**Context:** Finding 2.11 (Architect: `EspnConnector._fetch_per_team` swallows exceptions at DEBUG level) and 3.28 (PM: ESPN connector silently swallows all per-team exceptions) describe the identical issue. Pass 2 reclassified 2.11 as a duplicate of 3.28.

**Options:** N/A

**Recommendation:** Already Resolved

**Rationale:** Duplicate of 3.28. Fixed in Story 8.3 with tenacity retry + WARNING-level logging.

**Follow-up:** N/A

---

### 2.12 `get_data_dir()` `__file__`-Relative Path

**Decision needed:** Should `get_data_dir()` in `dashboard/lib/filters.py:56-58` be refactored away from `Path(__file__).resolve().parent.parent.parent / "data"`, which is fragile if the directory structure changes?

**Context:** The function uses `__file__`-relative path navigation (`Path(__file__).resolve().parent.parent.parent / "data"`) which is fragile if the directory structure moves. However, the dashboard directory structure has been stable since Epic 7.

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **A. Replace with environment variable or config** | Robust against directory restructuring; configurable for different deployments | Over-engineering for stable structure; adds configuration complexity |
| **B. Defer to post-MVP** | Low risk given stable directory structure; low priority | Fragility remains if structure ever changes |
| **C. Use `importlib.resources` or project root detection** | More Pythonic; survives directory moves | Moderate effort; may be tricky with dashboard being outside the package |

**Recommendation:** B. Defer to post-MVP

**Rationale:** Fragile if directory structure moves, but the dashboard directory structure has been stable since Epic 7. Low risk, low priority.

**Follow-up:** Add to Post-MVP Backlog

---

### 2.13 Dashboard Module-Level `_render_*()` Pattern

**Decision needed:** Should the dashboard pages be refactored away from the module-level `_render_*()` pattern, where all page logic runs on import?

**Context:** All page logic in `dashboard/pages/1_Lab.py:132` and `dashboard/pages/4_Pool_Scorer.py:248` runs at module level. This is standard Streamlit convention — Streamlit pages are scripts that re-execute on every interaction.

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **A. Wrap in `if __name__ == "__main__"` or page guard** | Prevents execution on non-Streamlit imports; more conventional Python | Breaks Streamlit's execution model; Streamlit relies on top-level execution |
| **B. Accept as-is** | Standard Streamlit convention; all Streamlit apps work this way | Surprising behavior for non-Streamlit imports (but these files are never imported elsewhere) |

**Recommendation:** B. Accept as-is

**Rationale:** This is standard Streamlit convention. All Streamlit apps work this way — code at module level runs on page navigation. Not a bug.

**Follow-up:** N/A

---

### 2.14 Undocumented Streamlit API

**Decision needed:** Should the leaderboard click-to-navigate feature be rewritten to avoid the undocumented `event.selection.rows` Streamlit API?

**Context:** `dashboard/pages/1_Lab.py:116-129` uses `event.selection.rows` with `# type: ignore[attr-defined]`. This API is undocumented but widely used in the Streamlit community. The risk is that a Streamlit upgrade breaks it.

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **A. Rewrite to use official Streamlit API** | Future-proof; no undocumented API risk | May not be possible — official Streamlit may not support this interaction pattern yet |
| **B. Defer to post-MVP** | Works currently; address if/when Streamlit breaks it | Risk of breakage on Streamlit upgrade |
| **C. Pin Streamlit version** | Eliminates upgrade risk | Blocks security patches and new features |

**Recommendation:** B. Defer to post-MVP

**Rationale:** The `event.selection.rows` API is undocumented but widely used in the Streamlit community. Risk is that a Streamlit upgrade breaks it. Low priority — will address if/when Streamlit breaks it.

**Follow-up:** Add to Post-MVP Backlog

---

### 2.15 Plotly Adapter API Changed from AC

**Decision needed:** Should the visualization API be changed from standalone functions back to model methods (e.g., `model.plot_calibration()`) as specified in the Epic AC?

**Context:** The Epic AC says `model.plot_calibration()` (methods on model objects). The implementation uses standalone functions (e.g., `plot_calibration(model, data)`). The Story documents this as a deliberate design decision — standalone functions were chosen over methods for composability.

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **A. Revert to model methods** | Matches Epic AC; more object-oriented API | Deliberately abandoned for good reasons (composability); couples visualization to model types |
| **B. Accept as-is** | Standalone functions are a deliberate, documented design decision; better composability | Deviates from Epic AC |

**Recommendation:** B. Accept as-is

**Rationale:** Standalone functions are a deliberate, documented design decision. The Story explicitly chose functions over methods for composability. Accept the documented deviation.

**Follow-up:** N/A

---

### 2.16 `st.spinner` Instead of `st.progress`

**Decision needed:** Should the Pool Scorer simulation use `st.progress` (determinate progress bar) instead of `st.spinner` (indeterminate spinner)?

**Context:** The Epic AC specifies `st.progress` bar for simulation progress. The implementation in `dashboard/pages/4_Pool_Scorer.py:92,162` uses `st.spinner()` (indeterminate). Already tracked as Post-MVP Backlog #19.

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **A. Implement `st.progress`** | Fulfills AC; better UX (users see how far along simulation is) | Requires passing progress callback through simulation engine; moderate effort |
| **B. Defer to post-MVP** | `st.spinner` works; progress bar is UX polish | AC not fulfilled; users don't know how long to wait |

**Recommendation:** B. Defer to post-MVP

**Rationale:** Already in Post-MVP Backlog #19. `st.spinner` works; `st.progress` is a UX polish.

**Follow-up:** N/A — remains in Post-MVP Backlog

---

### 2.17 Story 2.3 Open AI-Review Follow-ups

**Decision needed:** Should the Pandera schema validation and `iterrows` replacement from the Story 2.3 AI-review follow-ups be implemented?

**Context:** Story 2.3's AI review identified two follow-up items: (1) Add Pandera schema validation to KaggleConnector, and (2) Replace `iterrows()` with vectorized operations. Neither was completed. These are code quality improvements, not functional bugs — the connector works correctly.

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **A. Implement both** | Better data validation; consistent with project conventions | Code works correctly as-is; moderate effort for quality improvement |
| **B. Defer to post-MVP** | Focus on higher-value items; connector works correctly | Code quality improvements deferred indefinitely |
| **C. Implement Pandera only** | Schema validation has higher value (catches data issues); iterrows is lower priority | Partial completion of follow-up |

**Recommendation:** B. Defer to post-MVP

**Rationale:** Pandera schema validation and iterrows replacement in KaggleConnector are code quality improvements, not functional bugs. The connector works correctly.

**Follow-up:** Add to Post-MVP Backlog

---

### 2.18 Top-Level `__init__.py` Missing Re-exports

**Decision needed:** Should `from ncaa_eval import EloModel` work (by adding re-exports to `__init__.py`), or should the Style Guide be updated to document the actual import paths?

**Context:** The Style Guide claims `from ncaa_eval import EloModel` should work, but `src/ncaa_eval/__init__.py:1-3` does not re-export public API symbols. This is a documentation-vs-implementation gap.

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **A. Add re-exports to `__init__.py`** | Style Guide becomes accurate; convenient top-level imports | Triggers heavy module loading (sklearn, networkx, etc.) on any import from `ncaa_eval` |
| **B. Update Style Guide to document actual import paths** | No performance penalty; documentation matches reality | Less convenient imports; requires users to know submodule layout |
| **C. Use lazy `__getattr__` re-exports** | Convenient imports without eager loading | Complex to implement and debug |

**Recommendation:** B. Update Style Guide to document actual import paths

**Rationale:** Adding re-exports would trigger heavy module loading. Better to update the Style Guide to document the actual import paths rather than adding re-exports (which would trigger heavy module loading).

**Follow-up:** Create story in Epic 10

---

### 2.19 User Guide Documents Sliders As If They Exist

**Decision needed:** N/A — this item is already resolved.

**Context:** The user guide contained 50 lines of specification for game theory sliders that do not exist, with only a small `{note}` admonition. Pass 2 reclassified this from Category 2 to Category 3. Addressed in Story 8.4 with a prominent "NOT YET IMPLEMENTED" banner.

**Options:** N/A

**Recommendation:** Already Resolved

**Rationale:** Reclassified to Cat 3 in Pass 2. Addressed in Story 8.4 with "NOT YET IMPLEMENTED" banner.

**Follow-up:** N/A

---

### 2.20 No Data Post-Sync Validation

**Decision needed:** Should validation checks be added after data sync to catch silent data corruption (game count reasonableness, duplicate detection, team reference integrity)?

**Context:** No validation step checks data integrity after sync operations. Silent data corruption (missing games, duplicates, orphaned team references) could lead to incorrect model training and predictions. This is particularly important given the ESPN connector's history of partial data issues (3.28).

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **A. Implement post-sync validation** | Catches silent data corruption early; important given ESPN connector issues; improves data pipeline reliability | Moderate effort; requires defining "reasonable" thresholds for game counts per season |
| **B. Defer to post-MVP** | No work needed now; data has been manually verified to date | Silent corruption risk persists; harder to debug data issues retroactively |
| **C. Implement minimal validation (duplicate check only)** | Low effort; catches the most common corruption scenario | Misses game count and referential integrity checks |

**Recommendation:** A. Implement post-sync validation

**Rationale:** Post-sync validation (game count reasonableness, duplicate detection, team reference integrity) would catch silent data corruption. This is a data integrity improvement worth implementing, especially given the ESPN connector's history of issues.

**Follow-up:** Create story in Epic 10

---

### 2.21 `_make_season_df` Duplicated in Tests

**Decision needed:** Should the duplicated `_make_season_df` test helper be consolidated into a shared fixture?

**Context:** The same helper function is defined in both `tests/unit/test_evaluation_splitter.py:18` and `tests/unit/test_evaluation_backtest.py:28`. This is a minor code quality issue.

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **A. Consolidate now into shared conftest fixture** | DRY; easier maintenance | Minor effort for a minor issue; risks breaking existing tests if fixture behavior differs subtly |
| **B. Defer — consolidate when either file is next modified** | Natural cleanup opportunity; no dedicated effort | Duplication persists until next modification |

**Recommendation:** B. Defer — consolidate when either file is next modified

**Rationale:** Minor code quality issue — two test files share a small helper. Can be consolidated into a shared fixture when either file is next modified.

**Follow-up:** Add to Post-MVP Backlog

---

### P2-5 No Coverage Threshold

**Decision needed:** Should a minimum code coverage threshold be enforced in CI?

**Context:** CI runs `pytest --cov=src/ncaa_eval --cov-report=term-missing` but there is no `--cov-fail-under=XX` flag and no `fail_under` in `[tool.coverage.report]`. Coverage can silently regress without failing the build. The current coverage level has not been measured as a baseline.

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **A. Set threshold now (e.g., 80%)** | Prevents coverage regression; establishes quality baseline | May be too high or too low without measuring current coverage; could block legitimate PRs |
| **B. Measure current coverage first, then set threshold** | Data-driven threshold; avoids arbitrary numbers | Two-step process; defers the enforcement |
| **C. Defer to post-MVP** | No work needed now; no risk of blocking legitimate PRs | Coverage can silently regress |

**Recommendation:** C. Defer to post-MVP

**Rationale:** Need to measure current coverage level before setting a threshold. Setting an arbitrary threshold risks either being too low (useless) or too high (blocks legitimate PRs). Defer until a coverage audit is done.

**Follow-up:** Add to Post-MVP Backlog

---

### P2-6 Dashboard Excluded from Quality Gates

**Decision needed:** Should the dashboard package be included in mypy type checking and other quality gates?

**Context:** `dashboard/` is not included in mypy's scope (pyproject.toml), not in the noxfile typecheck session, and has 6 `# type: ignore` suppressions. The pre-commit mypy hook only checks `^(src/|tests/)`. The dashboard is the primary user-facing surface yet receives no static type checking. However, Streamlit has poor type stubs, making strict mypy impractical.

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **A. Add dashboard to mypy with relaxed config** | Catches import errors and basic type mismatches; some safety net | Streamlit's poor type stubs generate many false positives; requires per-module overrides |
| **B. Defer to post-MVP** | Avoids fighting Streamlit type stubs; focus on higher-value quality improvements | Dashboard continues with zero static analysis |
| **C. Add dashboard to ruff/linting only (not mypy)** | Catches style and basic code quality issues without type stub problems | Misses type errors; partial improvement |

**Recommendation:** B. Defer to post-MVP

**Rationale:** Streamlit has poor type stubs; strict mypy is impractical. A relaxed mypy config for `dashboard/` could catch basics but is a low-priority improvement.

**Follow-up:** Add to Post-MVP Backlog

---

### P3-20 Architecture Spec Stale

**Decision needed:** Should the architecture spec (`specs/05-architecture-fullstack.md`) be updated to match the current implementation, or marked as a historical document?

**Context:** Several spec claims don't match implementation: (1) `docs/specs/` path vs actual `/specs/`, (2) `dashboard/components/` vs actual `dashboard/lib/`, (3) Ingestion Engine uses `requests` vs actual `kaggle`/`cbbpy`, (4) Only 2 dashboard pages described vs actual 5. These are expected divergences from initial planning. Story 8.12 already added a historical-document banner.

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **A. Update spec to match implementation** | Accurate documentation; useful reference | Busywork; the code IS the spec now; spec will go stale again |
| **B. Accept as-is (historical document)** | Story 8.12 already added a historical-document banner; no additional work | Spec remains inaccurate (but labeled as such) |
| **C. Delete the spec entirely** | No stale documentation to maintain | Loses historical context of initial design decisions |

**Recommendation:** B. Accept as-is (historical document)

**Rationale:** Story 8.12 already added a historical-document banner. The spec served its purpose during initial design. Updating it to match implementation would be busywork — the code IS the spec now.

**Follow-up:** N/A

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
