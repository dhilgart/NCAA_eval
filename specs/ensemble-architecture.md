# Ensemble Architecture Design

**Status:** Approved design — 2026-03-09
**Origin:** PO decision session on po-decision-log-epic8.md (items 1.6 and Post-MVP ensemble)
**Implements:** Epic 10 stories + revised Story 9.2

---

## 1. Motivation

Two related gaps were identified during the Epic 8 audit and PO decision session:

1. **FeatureConfig is hardcoded inside `_setup_feature_server()`** and not exposed to library users. A user importing `ncaa_eval` programmatically cannot change the feature engineering configuration without editing library source code. Audit item 1.6.

2. **No ensemble modeling capability exists.** The Post-MVP Backlog contains a placeholder for ensemble/blending. The desired UX goes beyond simple weighted averaging — it requires learned, input-dependent weights (stacked generalization with a game-aware meta-learner).

These two gaps are coupled: ensemble modeling requires that each sub-model know its own feature requirements. FeatureConfig-as-model-concern is the foundational fix; ensemble modeling builds on top of it.

---

## 2. Part 1 — FeatureConfig as Model-Level Concern (prerequisite: revised Story 9.2)

### 2.1 Current state

`FeatureConfig` is constructed inside the private `_setup_feature_server(data_dir)` function in `src/ncaa_eval/cli/train.py:90-100` with hardcoded defaults. `run_training()` has no `feature_config` parameter. Library users cannot vary feature engineering without subclassing or monkey-patching.

Additionally, `FeatureConfig.calibration_method` is declared in the feature config but is never used by `StatefulFeatureServer` — calibration is a post-prediction transform, not a feature-production step. It belongs in `ModelConfig`, not `FeatureConfig`.

### 2.2 Target design

Each `Model` subclass declares its own `feature_config: FeatureConfig` as a class-level default, overridable through `__init__` kwargs:

```python
class XGBoostModel(Model):
    def __init__(
        self,
        *,
        batch_rating_types: tuple[BatchRatingType, ...] = ("srs",),
        graph_features_enabled: bool = False,
        ordinal_composite: OrdinalCompositeMethod | None = None,
        # ... other feature kwargs ...
        n_estimators: int = 100,
        max_depth: int = 4,
        # ... XGBoost hyperparams ...
    ) -> None:
        self.feature_config = FeatureConfig(
            batch_rating_types=batch_rating_types,
            graph_features_enabled=graph_features_enabled,
            ordinal_composite=ordinal_composite,
        )
        # ... store XGBoost hyperparams ...
```

`EloModel` uses a minimal config (no batch ratings, no ordinals needed — it reconstructs Game objects from metadata only):

```python
class EloModel(StatefulModel):
    def __init__(self, *, elo_config: EloConfig | None = None, ...) -> None:
        self.feature_config = FeatureConfig(
            sequential_windows=(),
            graph_features_enabled=False,
            batch_rating_types=(),
            ordinal_composite=None,
            elo_enabled=True,
            elo_config=elo_config,
        )
```

### 2.3 Changes to `run_training()`

`run_training()` reads `model.feature_config` to build the feature server instead of calling `_setup_feature_server(data_dir)` with hardcoded defaults:

```python
def run_training(model: Model, *, data_dir: Path, ...) -> ModelRun:
    server = _setup_feature_server(data_dir, model.feature_config)
    ...
```

The CLI `ncaa-eval train` remains unchanged in its external interface — it instantiates model classes (which carry their own default `feature_config`). Power users who want to vary feature config do so by passing different kwargs to the model constructor, not by passing a YAML flag.

### 2.4 `feature_config` serialization

`model.save(path)` must persist `feature_config` alongside model weights/parameters so that a loaded model knows exactly what feature columns it expects at inference time. Without this, a loaded `XGBoostModel` could be paired with a differently-configured feature server and silently receive wrong columns.

**Convention:** `save()` writes a `feature_config.json` sidecar file in the same directory as the model artifact. `load()` reads it and reconstructs `FeatureConfig`.

### 2.5 `feature_names_` post-fit convention

After `fit()`, every stateless model stores the feature column names it was trained on:

```python
self.feature_names_: list[str] = feat_cols  # set during fit()
```

This is the sklearn convention (`feature_names_in_`). It is required by ensemble models at inference time to route the right column slice from a superset DataFrame to each sub-model.

### 2.6 `calibration_method` relocation

Remove `calibration_method` from `FeatureConfig`. Add it to `ModelConfig` as an optional field. This is a correctness fix — calibration is applied to model outputs, not to feature computation.

---

## 3. Part 2 — StackedEnsemble Design

### 3.1 Architecture overview

`StackedEnsemble` is a stacked generalization model with a game-aware meta-learner:

- **Level 0 (base models):** Any number of `Model` instances (stateful or stateless), each with their own `feature_config`, trained independently via walk-forward cross-validation to produce out-of-fold (OOF) predictions.
- **Level 1 (meta-learner):** Any `Model` instance that learns to combine base model predictions. Its inputs are `[pred_base_0, pred_base_1, ..., seed_diff, is_tournament, loc_encoding, ...]` — the base model predictions plus a set of contextual game features. Because the meta-learner receives game-level context, it can learn input-dependent weights (trust XGBoost more for high-seed-differential games, trust Elo more for early-season neutral-site games, etc.).

### 3.2 Constructor

```python
@dataclass
class StackedEnsemble:
    base_models: list[Model]
    meta_learner: Model
    contextual_features: list[str] = field(
        default_factory=lambda: ["seed_diff", "is_tournament", "loc_encoding"]
    )
```

`contextual_features` is a list of column names from the minimal feature set. These columns are always present in any feature server output (they come from the SEED block and game metadata). They give the meta-learner the game context it needs to vary weights by game type without requiring any additional feature engineering.

### 3.3 `feature_config` property on `StackedEnsemble`

`StackedEnsemble` exposes a `feature_config` that is the **union** of all base model `feature_config`s, computed lazily. This is used by `run_training()` when it detects a `StackedEnsemble` and needs to know what seed/ordinal/rating blocks are needed to produce the `contextual_features` for the meta-learner.

In practice, `contextual_features` defaults to columns that are always present (`seed_diff` from the SEED block, `is_tournament` and `loc_encoding` from metadata), so the meta-learner's feature server needs only a minimal config with SEED enabled.

---

## 4. Part 3 — Training Flow

### 4.1 One-line UX

```python
ensemble = StackedEnsemble(
    base_models=[XGBoostModel(batch_rating_types=("srs",)), EloModel()],
    meta_learner=XGBoostModel(),
)
run_training(ensemble, data_dir=data_dir, start_year=2015, end_year=2024, output_dir=output_dir, model_name="my_ensemble")
```

### 4.2 `run_training()` dispatch

```python
def run_training(model: Model | StackedEnsemble, ...) -> ModelRun:
    if isinstance(model, StackedEnsemble):
        return _run_ensemble_training(model, ...)
    # existing leaf-model path
    server = _setup_feature_server(data_dir, model.feature_config)
    ...
```

### 4.3 `_run_ensemble_training()` steps

1. **OOF generation (per base model):**
   For each `base_model` in `ensemble.base_models`:
   - Build a feature server using `base_model.feature_config`
   - Run `run_backtest(base_model, server, seasons=all_seasons)` to produce walk-forward OOF predictions
   - Store OOF predictions indexed by `game_id`

2. **OOF alignment:**
   Join all base models' OOF predictions on `game_id` using an inner join. Games where any base model produced no prediction (e.g., due to all-NaN features) are dropped from the meta-training set. Log a warning if the inner join drops >5% of games.

3. **Meta-training set construction:**
   Build a minimal feature server with the config needed to produce `contextual_features`. Join the contextual feature columns onto the aligned OOF predictions by `game_id`. The resulting DataFrame has columns `[pred_base_0, pred_base_1, ..., seed_diff, is_tournament, loc_encoding, team_a_won]`.

4. **Meta-learner training:**
   `ensemble.meta_learner.fit(meta_X, meta_y)` where `meta_X` is the meta-training DataFrame and `meta_y` is `team_a_won`.

5. **Final base model retraining:**
   Re-train each base model on the full dataset (all seasons) using its own feature server. This produces the final base models used at inference time.

6. **Artifact persistence:**
   Save each base model, the meta-learner, and the ensemble manifest (ordered list of base model names, `contextual_features`, run IDs of base model backtest runs used to produce the OOF predictions). The manifest enables stale-meta-learner detection in future runs.

### 4.4 Total compute cost

For K base models and S seasons: `(S + 1) × sum(base model training time)`. The `+1` comes from the final full-data retraining. This is standard stacking overhead and is proportional to the number of base models, not combinatorially worse.

---

## 5. Part 4 — Inference Interface

### 5.1 Two inference modes

**Backtest / evaluation context** (`predict_proba(X: pd.DataFrame)`):
The ensemble receives a pre-built superset DataFrame. For each base model:
- Stateless: `base_model.predict_proba(X[base_model.feature_names_])`
- Stateful: `base_model.predict_proba(X)` (uses `team_a_id`/`team_b_id` from metadata; ignores feature columns)

Then assemble meta-learner input from the base predictions + `X[contextual_features]` and call `meta_learner.predict_proba(meta_X)`.

**Live bracket prediction** (`predict_bracket(data_dir: Path, season: int) -> pd.DataFrame`):
The ensemble independently builds a current-season feature matrix per base model, generates base model predictions, assembles the meta-learner input with contextual features, and returns a probability matrix for all team matchups.

```python
def predict_bracket(self, data_dir: Path, season: int) -> pd.DataFrame:
    # For each base model: build feature server, serve current season, predict
    # Assemble meta-input, call meta_learner.predict_proba()
    # Return 64×64 probability matrix
    ...
```

`predict_bracket()` is the intended interface for the Streamlit dashboard and Kaggle submission export. `predict_proba(X)` is used internally by the backtest engine.

### 5.2 Meta-learner input schema at inference

The meta-learner's input column order must exactly match its training column order. `StackedEnsemble.save()` persists the ordered column list. `predict_proba()` and `predict_bracket()` both reconstruct the meta-input DataFrame in that exact order before calling `meta_learner.predict_proba()`.

---

## 6. Implementation Dependencies

```
Story 9.2 (revised): feature_config on Model ABC + serialization + feature_names_ post-fit
    ↓
Story 10.1: StackedEnsemble class + OOF training pipeline (_run_ensemble_training)
    ↓
Story 10.2: Ensemble inference (predict_bracket + predict_proba routing)
    ↓
Story 10.3: Dashboard + model registry integration (StackedEnsemble visible in leaderboard)
    ↓
Story 10.4: Ensemble tutorial notebook
```

Story 9.2 must be merged before any Epic 10 story begins. Stories 10.1–10.4 are sequential (each builds on the previous).

---

## 7. Open Questions (resolved during PO session 2026-03-09)

| Question | Decision |
|---|---|
| Should FeatureConfig stay external or move to the model? | Embedded in model as init kwargs → FeatureConfig |
| CLI flag approach (YAML path vs individual flags)? | Model-level; CLI is a thin wrapper on model constructors |
| Superset feature DataFrame vs per-sub-model feature generation? | Per-sub-model for training; at inference, stateless subs use feature_names_ for routing from a pre-built DataFrame; live bracket uses predict_bracket() |
| Fixed blend weights vs learned weights? | Learned (meta-learner) |
| Uniform weights vs input-dependent weights? | Input-dependent via game-context features as meta-learner inputs |
| End-to-end ensemble training or train-then-compose? | End-to-end via run_training() dispatcher; StackedEnsemble is self-contained |
| Inference interface for hypothetical bracket games? | predict_bracket(data_dir, season) — explicit data dependency |
| calibration_method in FeatureConfig? | Move to ModelConfig — it's a model concern, not a feature-server concern |
