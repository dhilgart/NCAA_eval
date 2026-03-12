# Story 10.2: Ensemble Inference Interface

Status: done

## Story

As a data scientist,
I want to generate bracket predictions and evaluate an ensemble on historical data using the same interfaces I use for single models,
so that ensembles compose transparently with the existing evaluation and bracket-generation infrastructure.

## Acceptance Criteria

1. **predict_proba routing** — Given a trained `StackedEnsemble` and a pre-built feature DataFrame `X`, when `ensemble.predict_proba(X)` is called, then:
   - For each **stateless** base model: `base_model.predict_proba(X[base_model.feature_names_])` is called using the stored post-fit feature name list
   - For each **stateful** base model: `base_model.predict_proba(X)` is called (stateful models use `team_a_id`/`team_b_id` from metadata and ignore feature columns)
   - Base model predictions and `X[contextual_features]` are assembled into a meta-input DataFrame in the column order recorded during training (`meta_column_order` from manifest)
   - `meta_learner.predict_proba(meta_X)` returns the final ensemble probability

2. **predict_bracket** — Given a trained `StackedEnsemble` and a `data_dir` containing current-season data, when `ensemble.predict_bracket(data_dir, season)` is called, then:
   - For each base model, a feature server is built from `base_model.feature_config` and current-season features are served
   - Base model predictions for all possible team matchups are generated
   - The meta-input is assembled with contextual features from the current season
   - A probability matrix (indexed by team_id pairs) is returned, suitable for passing to the Monte Carlo bracket simulator

3. **Column order enforcement** — In both modes, the meta-learner input column order exactly matches the order recorded in the ensemble manifest (`meta_column_order`), and a `ValueError` is raised if any required column is missing

4. **CLI predict integration** — The `NotImplementedError` guard in `cli/predict.py` is replaced with working ensemble prediction that routes through `predict_proba` or `predict_bracket` as appropriate

5. **ProbabilityProvider integration** — A new `EnsembleProvider` (or equivalent mechanism) implements the `ProbabilityProvider` protocol so that `build_probability_matrix()` and the Monte Carlo simulator work identically to single-model mode

## Tasks / Subtasks

- [x] Task 1: Add `meta_column_order` to `StackedEnsemble` and wire it through `load()` (AC: #3)
  - [x] 1.1 Add `meta_column_order: list[str]` field to `StackedEnsemble` (default empty list; populated during training or load)
  - [x] 1.2 Update `StackedEnsemble.load()` to read `meta_column_order` from manifest.json and set it on the instance
  - [x] 1.3 Update `_run_ensemble_training` in `cli/train.py` to set `ensemble.meta_column_order` after building the meta-training set (before save, so it's available for inference immediately after training without reload)
  - [x] 1.4 Add unit test: loaded ensemble has correct `meta_column_order`

- [x] Task 2: Implement `StackedEnsemble.predict_proba(X)` (AC: #1, #3)
  - [x] 2.1 Add `predict_proba(self, X: pd.DataFrame) -> pd.Series` method to `StackedEnsemble`
  - [x] 2.2 Route stateless base models through `X[base_model.feature_names_]`; stateful through `X` directly
  - [x] 2.3 Assemble `meta_X` DataFrame with columns in `self.meta_column_order`; raise `ValueError` if any column is missing
  - [x] 2.4 Call `self.meta_learner.predict_proba(meta_X)` and return result
  - [x] 2.5 Unit tests: correct predictions with mixed stateful/stateless base models; column order enforcement; ValueError on missing columns

- [x] Task 3: Implement `StackedEnsemble.predict_bracket(data_dir, season)` (AC: #2, #3)
  - [x] 3.1 Add `predict_bracket(self, data_dir: Path, season: int) -> pd.DataFrame` method
  - [x] 3.2 For each base model: build feature server from `base_model.feature_config`, serve season features, generate predictions
  - [x] 3.3 Handle stateful models: they need to be fit on prior seasons' games before predicting current-season matchups (use `_setup_feature_server` + `fit` on all prior data)
  - [x] 3.4 For all C(n,2) team pairings: assemble meta-input (base predictions + contextual features) in `meta_column_order`
  - [x] 3.5 Return n×n probability matrix as DataFrame indexed by team_id
  - [x] 3.6 Unit test: verify matrix shape, symmetry (`P[a,b] + P[b,a] ≈ 1`), zero diagonal

- [x] Task 4: Replace `NotImplementedError` in `cli/predict.py` (AC: #4, #5)
  - [x] 4.1 Remove the `StackedEnsemble` guard from `build_predictions()`
  - [x] 4.2 Add ensemble prediction path: load ensemble, call `predict_bracket(data_dir, season)` to get probability matrix, convert to prediction rows
  - [x] 4.3 Ensure CSV output format matches single-model output (`season,team_a_id,team_b_id,pred_win_prob`)
  - [x] 4.4 Integration test: CLI predict with a mock ensemble produces valid CSV

- [x] Task 5: Quality gates (AC: all)
  - [x] 5.1 `ruff check .` clean
  - [x] 5.2 `mypy --strict src/ncaa_eval tests` clean
  - [x] 5.3 Full `pytest` suite passes (existing + new tests)
  - [x] 5.4 No regressions in single-model prediction paths

## Dev Notes

### Critical Design Decisions from Story 10.1

- **`StackedEnsemble` is NOT a `Model` subclass** — it's a standalone `@dataclass`. The `predict_proba` method you add is NOT an override of `Model.predict_proba`; it's a new method with the same name following the same signature convention.
- **`_EnsembleSentinel`** in the registry is a placeholder only. Real loading goes through `StackedEnsemble.load()`. `RunStore.load_model()` already handles the `model_type == "ensemble"` dispatch.
- **`meta_column_order`** is persisted in `manifest.json` (augmented after `ensemble.save()` in `_run_ensemble_training`). The manifest is at `<model_dir>/manifest.json`. Load it and set it on the `StackedEnsemble` instance during `load()`.

### Stateful vs Stateless Model Routing at Inference

**Stateless models** (XGBoost, LogisticRegression):
- Have `feature_names_: list[str]` set during `fit()` and persisted via `feature_names.json`
- At inference: `base_model.predict_proba(X[base_model.feature_names_])` — slice the superset DataFrame to the exact columns this model was trained on
- These models return `pd.Series` of probabilities

**Stateful models** (Elo):
- Use `team_a_id`/`team_b_id` from the DataFrame metadata columns — they ignore feature columns entirely
- At inference: `base_model.predict_proba(X)` — pass the full DataFrame (they extract metadata via `itertuples`)
- `StatefulModel.predict_proba` calls `_predict_one(row.team_a_id, row.team_b_id)` per row

### Meta-Learner Input Assembly

The meta-learner was trained on a DataFrame with columns in this exact order (from `meta_column_order` in manifest):
```
["pred_base_0", "pred_base_1", ..., "seed_diff", "is_tournament", "loc_encoding"]
```

At inference, you must:
1. Collect base model predictions into columns `pred_base_0`, `pred_base_1`, etc. (index matches `base_models` list order)
2. Extract contextual features from `X` (for `predict_proba`) or from a feature server (for `predict_bracket`)
3. Assemble a DataFrame with columns in exactly `self.meta_column_order`
4. Validate: if any column in `meta_column_order` is missing, raise `ValueError` with a clear message listing the missing columns

### predict_bracket Architecture

`predict_bracket(data_dir, season)` must handle the full pipeline for hypothetical matchups:

1. **Discover teams**: Load tournament seeds for the season from the repository to get the set of team IDs
2. **Per base model**:
   - Build a `StatefulFeatureServer` from `base_model.feature_config` (use `_setup_feature_server(data_dir, base_model.feature_config)`)
   - **Stateless models**: Serve batch features for the season, then for each C(n,2) pairing, slice the features and predict. Alternatively, construct synthetic feature rows for each pairing if batch features don't cover hypothetical matchups.
   - **Stateful models**: Fit on all prior seasons' games (the base model is already fitted on full data from training step 5), then call `predict_matchup(team_a_id, team_b_id)` directly via the Elo internal API
3. **Assemble meta-input**: For each pairing, combine base predictions + contextual features
4. **Return**: n×n probability matrix as `pd.DataFrame` with team_id index/columns

**Important**: The base models loaded from disk are already retrained on the full dataset (training step 5). Stateful models (Elo) have internal state (ratings) from fitting on all historical games. They can directly predict matchups without re-fitting. Stateless models need feature rows to predict — these come from the feature server.

### Existing Code to Reuse (DO NOT Reinvent)

| Need | Existing Solution | Location |
|------|-------------------|----------|
| Build feature server | `_setup_feature_server(data_dir, feature_config)` | `cli/train.py` |
| Probability matrix construction | `build_probability_matrix(provider, team_ids, context)` | `evaluation/providers.py` |
| Matchup context | `MatchupContext(season, day_num, is_neutral)` | `evaluation/bracket.py` |
| Elo pairwise predictions | `EloProvider(model)` | `evaluation/providers.py` |
| Matrix-based provider | `MatrixProvider(matrix, team_ids)` | `evaluation/providers.py` |
| Tournament team discovery | `ParquetRepository.get_seeds(season)` | `ingest/repository.py` |
| Neutral day number | `KAGGLE_NEUTRAL_DAY_NUM` | `evaluation/kaggle_export.py` |
| Feature column filtering | `feature_cols(df)` → non-metadata columns | `evaluation/__init__.py` |

### predict_proba Implementation Pattern

```python
def predict_proba(self, X: pd.DataFrame) -> pd.Series:
    """Route features through base models and meta-learner."""
    base_preds: dict[str, pd.Series] = {}
    for i, base_model in enumerate(self.base_models):
        col_name = f"pred_base_{i}"
        if isinstance(base_model, StatefulModel):
            base_preds[col_name] = base_model.predict_proba(X)
        else:
            base_preds[col_name] = base_model.predict_proba(
                X[base_model.feature_names_]
            )

    # Assemble meta-input in training column order
    meta_parts = {**base_preds}
    for feat in self.contextual_features:
        if feat in X.columns:
            meta_parts[feat] = X[feat]

    meta_df = pd.DataFrame(meta_parts, index=X.index)

    # Validate column order
    missing = [c for c in self.meta_column_order if c not in meta_df.columns]
    if missing:
        msg = f"Missing meta-learner input columns: {missing}"
        raise ValueError(msg)

    meta_X = meta_df[self.meta_column_order]
    return self.meta_learner.predict_proba(meta_X)
```

### File Locations for Changes

| File | Action | Description |
|------|--------|-------------|
| `src/ncaa_eval/model/ensemble.py` | MODIFY | Add `meta_column_order` field, `predict_proba()`, `predict_bracket()` methods; update `load()` |
| `src/ncaa_eval/cli/train.py` | MODIFY | Set `ensemble.meta_column_order` after building meta-training set |
| `src/ncaa_eval/cli/predict.py` | MODIFY | Replace `NotImplementedError` with working ensemble prediction path |
| `tests/unit/test_model_ensemble.py` | MODIFY | Add tests for `predict_proba`, `predict_bracket`, column validation |
| `tests/unit/test_cli_predict.py` | MODIFY | Add ensemble prediction integration test (or create if not exists) |

### Testing Strategy

**Unit tests** (mock base models and meta-learner):
- `predict_proba` returns correct shape and values with known mock predictions
- Stateless routing: verify `base_model.predict_proba` called with sliced DataFrame
- Stateful routing: verify `base_model.predict_proba` called with full DataFrame
- Column order: meta-learner receives columns in exact `meta_column_order`
- Missing column `ValueError`: remove a contextual feature from X and verify error
- `predict_bracket` returns correct matrix shape and symmetry property

**Integration tests** (with real model stubs):
- Round-trip: train ensemble → save → load → `predict_proba` produces same results
- CLI predict: mock ensemble run → `build_predictions` returns valid CSV

### Story 10.1 Learnings to Apply

- `caplog` tests need logger propagation re-enabled: `logging.getLogger("ncaa_eval").propagate = True` in test fixtures
- Use `copy.deepcopy()` when models need isolated state (already done for OOF but relevant for any test that mutates model state)
- Keep helper functions extracted to stay under C901 complexity threshold
- `_EnsembleSentinel` is `pragma: no cover` — don't test it

### Anti-Patterns to Avoid

- **DO NOT** make `StackedEnsemble` a `Model` subclass — it deliberately avoids the ABC to not duplicate the full `Model` interface (it has no `fit`, no `get_config` dispatch through the standard training path)
- **DO NOT** re-implement probability matrix construction — use `build_probability_matrix()` from `evaluation/providers.py`
- **DO NOT** re-implement feature server setup — use `_setup_feature_server()` from `cli/train.py`
- **DO NOT** use `iterrows` for prediction assembly — use vectorized pandas operations
- **DO NOT** hardcode `meta_column_order` — always read from the manifest/instance field

### Project Structure Notes

- All new code goes in existing files — no new modules needed
- `from __future__ import annotations` required in all modified files (already present)
- `mypy --strict` applies to all files in `src/` and `tests/`
- Google-style docstrings required for public methods

### References

- [Source: specs/ensemble-architecture.md#5] — Inference interface design (predict_proba and predict_bracket modes)
- [Source: specs/ensemble-architecture.md#5.2] — Meta-learner input schema at inference
- [Source: _bmad-output/planning-artifacts/epics.md] — Epic 10, Story 10.2 acceptance criteria
- [Source: _bmad-output/implementation-artifacts/10-1-stacked-ensemble-oof-training-pipeline.md] — Previous story dev notes and file list
- [Source: src/ncaa_eval/model/ensemble.py] — StackedEnsemble class, save/load, feature_config union
- [Source: src/ncaa_eval/cli/train.py] — _run_ensemble_training, meta_column_order persistence
- [Source: src/ncaa_eval/cli/predict.py] — NotImplementedError guard to replace
- [Source: src/ncaa_eval/evaluation/providers.py] — ProbabilityProvider protocol, MatrixProvider, EloProvider, build_probability_matrix
- [Source: src/ncaa_eval/evaluation/bracket.py] — MatchupContext, BracketStructure

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

- Pre-commit hook caught unused `type: ignore[type-arg]` on `predict_proba` return type — mypy infers `pd.Series` fine without the annotation
- Pre-commit hook caught `Model` has no attribute `feature_names_` — used `type: ignore[attr-defined]` since stateless models set this dynamically during `fit()`
- Ruff format auto-fixed parenthesization in `_build_ensemble_predictions` tuple append

### Completion Notes List

- **Task 1**: Added `meta_column_order: list[str]` field to `StackedEnsemble` dataclass with default empty list. Updated `save()` to persist in manifest.json, `load()` to read from manifest (with `.get()` fallback for backward compatibility), and `_run_ensemble_training` to set it on the instance before save. 3 unit tests cover default, round-trip, and legacy manifest scenarios.
- **Task 2**: Implemented `predict_proba(X)` that routes stateful base models through full DataFrame and stateless models through `X[feature_names_]`. Assembles meta-input in `meta_column_order`, validates completeness, and delegates to meta-learner. 4 unit tests cover stateless routing, stateful routing, column order enforcement, and missing column ValueError.
- **Task 3**: Implemented `predict_bracket(data_dir, season)` with extracted helper functions for complexity management. Stateful base models use `EloProvider` + `build_probability_matrix()`. Stateless base models use synthetic feature row construction from per-team season profiles extracted from the feature server. Contextual features (seed_diff, is_tournament, loc_encoding) are computed for all C(n,2) pairs. Returns n×n DataFrame indexed by team_id. 1 unit test verifies shape, symmetry, and zero diagonal.
- **Task 4**: Replaced `NotImplementedError` in `build_predictions()` with `_build_ensemble_predictions()` that calls `ensemble.predict_bracket()` and converts the probability matrix to CSV-compatible rows. CSV output format matches single-model output. 1 integration test verifies valid CSV output through the CLI.
- **Task 5**: All quality gates pass — `ruff check .` clean, `mypy --strict` clean (105 files), `pytest` 1163 passed / 0 failed / 1 skipped. No regressions in single-model prediction paths.

### File List

- `src/ncaa_eval/model/ensemble.py` — MODIFIED: Added `meta_column_order` field, `predict_proba()`, `predict_bracket()` methods; updated `save()`/`load()`; added bracket helper functions; fixed tournament-only team discovery, deterministic iteration, length validation, `predict_bracket` `Raises:` docstring
- `src/ncaa_eval/cli/train.py` — MODIFIED: Set `ensemble.meta_column_order` after building meta-training set
- `src/ncaa_eval/cli/predict.py` — MODIFIED: Replaced `NotImplementedError` with working ensemble prediction path via `_build_ensemble_predictions()`; added probability clamp
- `src/ncaa_eval/evaluation/providers.py` — MODIFIED: Added `EnsembleProvider` class implementing `ProbabilityProvider` (AC #5)
- `src/ncaa_eval/evaluation/__init__.py` — MODIFIED: Export `EnsembleProvider`
- `tests/unit/test_model_ensemble.py` — MODIFIED: Added `TestMetaColumnOrder` (3 tests), `TestPredictProba` (4 tests), `TestPredictBracket` (2 tests — added missing column ValueError test)
- `tests/unit/test_cli_predict.py` — MODIFIED: Added `TestEnsemblePredict` (1 test)
- `_bmad-output/implementation-artifacts/sprint-status.yaml` — MODIFIED: Story status updated
- `_bmad-output/planning-artifacts/template-requirements.md` — MODIFIED: Added 3 new patterns from code review

## Change Log

- 2026-03-12: Implemented ensemble inference interface — `predict_proba()`, `predict_bracket()`, CLI predict integration, meta_column_order persistence. 9 new tests added. All quality gates pass.
- 2026-03-12: Code review fixes — Added `EnsembleProvider` (AC #5 completion); fixed tournament-only team discovery in `_discover_team_ids`; fixed non-deterministic set iteration; added context feature length validation; added probability clamp in CLI ensemble path; added `predict_bracket` missing-column ValueError test; added `Raises:` docstring section. 1 additional test. 1164 passed / 0 failed.
