# Story 6.1: Implement Metric Library (scikit-learn + numpy)

Status: done

## Story

As a data scientist,
I want a metric library computing Log Loss, Brier Score, ROC-AUC, ECE, and reliability diagram data,
So that I can evaluate model quality across multiple dimensions using vectorized operations.

## Acceptance Criteria

1. **Given** a set of model predictions (probabilities) and actual outcomes, **When** the developer calls `log_loss(y_true, y_prob)`, **Then** Log Loss is computed via `sklearn.metrics.log_loss` and returned as a `float`.
2. **Given** predictions and outcomes, **When** the developer calls `brier_score(y_true, y_prob)`, **Then** Brier Score is computed via `sklearn.metrics.brier_score_loss` and returned as a `float`.
3. **Given** predictions and outcomes, **When** the developer calls `roc_auc(y_true, y_prob)`, **Then** ROC-AUC is computed via `sklearn.metrics.roc_auc_score` and returned as a `float`.
4. **Given** predictions and outcomes, **When** the developer calls `expected_calibration_error(y_true, y_prob)`, **Then** ECE is computed using numpy vectorized operations (bin predictions, compute per-bin weighted absolute accuracy-confidence gap) and returned as a `float`.
5. **Given** predictions and outcomes, **When** the developer calls `reliability_diagram_data(y_true, y_prob)`, **Then** bin data is generated using `sklearn.calibration.calibration_curve` plus numpy for additional binning statistics (bin counts, bin edges), returned as a structured result.
6. All metric functions accept `npt.NDArray[np.float64]` inputs and return `float` (scalars) or `npt.NDArray[np.float64]` (reliability diagram arrays).
7. No Python `for` loops are used in any metric calculation — vectorization enforced per NFR1.
8. Each metric function is covered by unit tests with known expected values (hand-computed or reference library cross-checks).
9. Edge cases are handled: perfect predictions (loss=0), all-same-class (AUC undefined — raise `ValueError`), single prediction, empty arrays.

## Tasks / Subtasks

- [x] Task 1: Create `src/ncaa_eval/evaluation/metrics.py` (AC: #1–#7)
  - [x] 1.1 `log_loss(y_true, y_prob) -> float` — thin wrapper around `sklearn.metrics.log_loss`
  - [x] 1.2 `brier_score(y_true, y_prob) -> float` — thin wrapper around `sklearn.metrics.brier_score_loss`
  - [x] 1.3 `roc_auc(y_true, y_prob) -> float` — thin wrapper around `sklearn.metrics.roc_auc_score`
  - [x] 1.4 `expected_calibration_error(y_true, y_prob, *, n_bins: int = 10) -> float` — custom numpy vectorized implementation
  - [x] 1.5 `reliability_diagram_data(y_true, y_prob, *, n_bins: int = 10) -> ReliabilityData` — wraps `sklearn.calibration.calibration_curve` with extra bin statistics
  - [x] 1.6 Define `ReliabilityData` dataclass for structured return from reliability diagram
  - [x] 1.7 Input validation: check array lengths match, non-empty, probabilities in [0,1]
- [x] Task 2: Export public API from `src/ncaa_eval/evaluation/__init__.py` (AC: #6)
  - [x] 2.1 Add imports and `__all__` for all metric functions and `ReliabilityData`
- [x] Task 3: Create `tests/unit/test_evaluation_metrics.py` (AC: #8, #9)
  - [x] 3.1 Test `log_loss` with known expected values
  - [x] 3.2 Test `brier_score` with known expected values
  - [x] 3.3 Test `roc_auc` with known expected values
  - [x] 3.4 Test `expected_calibration_error` with hand-computed expected values
  - [x] 3.5 Test `reliability_diagram_data` output structure and values
  - [x] 3.6 Test edge case: perfect predictions (all correct)
  - [x] 3.7 Test edge case: all-same-class raises `ValueError` for `roc_auc`
  - [x] 3.8 Test edge case: single prediction
  - [x] 3.9 Test edge case: empty arrays raise `ValueError`
  - [x] 3.10 Test edge case: mismatched array lengths raise `ValueError`
  - [x] 3.11 Test edge case: probabilities outside [0,1] raise `ValueError`

## Dev Notes

### Design Reference

- [Source: _bmad-output/planning-artifacts/epics.md — Epic 6, Story 6.1 (lines 704–722)]
- [Source: _bmad-output/planning-artifacts/template-requirements.md — NFR1 vectorization mandate]

### Critical Implementation Constraints

1. **Library-First Rule**: Use `sklearn.metrics` for Log Loss, Brier Score, ROC-AUC — do NOT reimplement. Only ECE requires custom numpy code (sklearn does not provide it).
2. **Vectorization (NFR1)**: Zero Python `for` loops in metric calculations. ECE must use `np.digitize` + `np.bincount` or equivalent vectorized binning — no iteration over bins.
3. **`from __future__ import annotations`** required in all Python files (Ruff enforcement).
4. **`mypy --strict`** mandatory — use `npt.NDArray[np.float64]` from `numpy.typing` for array type hints.
5. **Mutation testing**: `src/ncaa_eval/evaluation/` is explicitly included in `[tool.mutmut]` paths — tests must catch subtle mutations (e.g., off-by-one in ECE bin counts, wrong sklearn function).

### File Structure

```
src/ncaa_eval/evaluation/
├── __init__.py          # Existing — add re-exports
├── metrics.py           # NEW — all metric functions + ReliabilityData
```

```
tests/unit/
├── test_evaluation_metrics.py  # NEW — comprehensive metric tests
```

### Dependencies

All required libraries are already in `pyproject.toml`:
- `numpy` — vectorized ECE computation
- `scikit-learn` — `sklearn.metrics.log_loss`, `sklearn.metrics.brier_score_loss`, `sklearn.metrics.roc_auc_score`, `sklearn.calibration.calibration_curve`

No new dependencies needed.

### Existing Codebase Context — DO NOT Reimplement

- **`src/ncaa_eval/transform/calibration.py`**: Provides `IsotonicCalibrator` and `SigmoidCalibrator` for probability calibration. These are *consumers* of metrics (calibrate then measure), not metrics themselves. Do not duplicate calibration logic.
- **`src/ncaa_eval/model/base.py`**: Models produce `pd.Series` of probabilities via `predict_proba()`. Story 6.2/6.3 will convert these to numpy arrays before passing to metrics. Metric functions should accept raw numpy arrays, not pandas objects.
- **sklearn import pattern**: Follow the lazy-import pattern from `calibration.py` — import sklearn inside function bodies with `# type: ignore[import-untyped]` to avoid top-level import cost and mypy errors on untyped sklearn stubs.

### ECE Algorithm Specification

ECE (Expected Calibration Error) is not available in scikit-learn. Implement using vectorized numpy:

1. Digitize predictions into `n_bins` equal-width bins on [0, 1] using `np.digitize`
2. For each bin (vectorized via `np.bincount` or boolean indexing):
   - `acc_b` = mean of `y_true` in the bin (fraction of positives)
   - `conf_b` = mean of `y_prob` in the bin (average predicted probability)
   - `weight_b` = count of samples in the bin / total samples
3. ECE = sum over bins of `weight_b * |acc_b - conf_b|`
4. Empty bins contribute 0 (skip via mask, do NOT loop).

### ReliabilityData Specification

Use a `dataclasses.dataclass` (frozen=True) to return structured reliability diagram data:

```python
@dataclasses.dataclass(frozen=True)
class ReliabilityData:
    fraction_of_positives: npt.NDArray[np.float64]  # from calibration_curve
    mean_predicted_value: npt.NDArray[np.float64]    # from calibration_curve
    bin_counts: npt.NDArray[np.int64]                # samples per bin
    n_bins: int                                       # requested bins
```

### Edge Case Handling

| Edge Case | Log Loss | Brier | ROC-AUC | ECE | Reliability |
|---|---|---|---|---|---|
| Empty arrays | `ValueError` | `ValueError` | `ValueError` | `ValueError` | `ValueError` |
| Single prediction | Valid (compute) | Valid (compute) | `ValueError` (needs 2+ classes) | Valid (compute) | Valid (1 bin) |
| All-same-class | Valid (compute) | Valid (compute) | `ValueError` (AUC undefined) | Valid (compute) | Valid (compute) |
| Perfect predictions | 0.0 | 0.0 | 1.0 | 0.0 | Perfect diagonal |
| Probs outside [0,1] | `ValueError` | `ValueError` | `ValueError` | `ValueError` | `ValueError` |

### Testing Patterns

Follow the test patterns established in `tests/unit/test_calibration.py`:
- Class-based test organization: `TestLogLoss`, `TestBrierScore`, etc.
- Known expected values: compute by hand or verify against direct sklearn calls
- Edge case coverage with `pytest.raises(ValueError, match="...")`
- NumPy `np.isclose()` / `pytest.approx()` for floating-point assertions
- Use `np.random.default_rng(seed)` for reproducible test data generation

### Previous Story Learnings (Epic 5)

- **`# type: ignore[import-untyped]`** required for sklearn imports (no type stubs)
- **`frozen=True` dataclass**: Use for `ReliabilityData` — immutable return values prevent caller mutation bugs
- **Empty DataFrame/array guards**: Validate inputs before passing to sklearn (sklearn may give unhelpful errors on empty input)
- **Property-based testing**: Consider `@pytest.mark.property` Hypothesis tests for invariants (e.g., ECE in [0, 1], Brier in [0, 1])

### Project Structure Notes

- `src/ncaa_eval/evaluation/` already exists with empty `__init__.py` — this is the designated module for all evaluation functionality
- New file `metrics.py` follows the pattern of `transform/calibration.py` (sibling module, same numpy/sklearn patterns)
- No conflicts with existing code — `evaluation/` is currently unused

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Epic 6, Story 6.1]
- [Source: src/ncaa_eval/transform/calibration.py — sklearn import pattern, numpy typing, edge case handling]
- [Source: tests/unit/test_calibration.py — test organization pattern, assertion patterns]
- [Source: src/ncaa_eval/model/base.py — Model.predict_proba() returns pd.Series of probabilities]
- [Source: pyproject.toml — numpy, scikit-learn already declared as dependencies]
- [Source: _bmad-output/planning-artifacts/template-requirements.md — NFR1 vectorization, Library-First Rule, mypy --strict, mutation testing]

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

- sklearn `log_loss` raises `ValueError` when `y_true` contains only one class and `labels` is not specified. Fixed by passing `labels=[0, 1]` explicitly.
- sklearn `brier_score_loss` and `roc_auc_score` have type stubs in current version — `# type: ignore[import-untyped]` not needed (removed to satisfy `mypy --strict` unused-ignore check). `log_loss` and `calibration_curve` still need the ignore.

### Completion Notes List

- Implemented 5 metric functions (`log_loss`, `brier_score`, `roc_auc`, `expected_calibration_error`, `reliability_diagram_data`) plus `ReliabilityData` frozen dataclass
- All sklearn wrappers use lazy imports inside function bodies (consistent with `calibration.py` pattern)
- ECE uses fully vectorized numpy: `np.digitize` + `np.bincount` with weighted sums — zero Python `for` loops
- Input validation via shared `_validate_inputs()` helper: checks non-empty, matching lengths, probabilities in [0, 1]
- `roc_auc` adds explicit check for single-class `y_true` with clear error message (before sklearn's less informative error)
- 48 unit tests covering known values, edge cases (empty, single, all-same-class, perfect, out-of-range), and structural properties
- All 545 tests pass (0 regressions), ruff clean, mypy --strict clean

### Change Log

- 2026-02-23: Implemented metric library with all 5 metric functions, ReliabilityData dataclass, input validation, public API exports, and 48 unit tests
- 2026-02-23: Code review fixes — added bin_edges to ReliabilityData (AC5 compliance), binary y_true validation to _validate_inputs, n_bins>=1 validation to ECE and reliability_diagram_data, corrected misleading comment on ECE binning; 7 new tests added (55 total, 552 suite-wide)
- 2026-02-23: Code review round 2 fixes — added .copy() for all numpy array fields in ReliabilityData constructor (frozen=True immutability contract); added test_non_binary_y_true_roc_auc (test coverage symmetry); updated __init__.py docstring from "Metrics and simulation module" to "Evaluation metrics module"; 1 new test (56 total)

### File List

- src/ncaa_eval/evaluation/metrics.py (NEW — code review: added bin_edges to ReliabilityData, binary y_true validation, n_bins>=1 validation, corrected misleading comment; code review round 2: .copy() for all array fields in ReliabilityData)
- src/ncaa_eval/evaluation/__init__.py (MODIFIED — updated docstring)
- tests/unit/test_evaluation_metrics.py (NEW — code review: 7 tests; code review round 2: test_non_binary_y_true_roc_auc)
- _bmad-output/implementation-artifacts/6-1-implement-metric-library-scikit-learn-numpy.md (MODIFIED)
- _bmad-output/implementation-artifacts/sprint-status.yaml (MODIFIED)
- _bmad-output/planning-artifacts/template-requirements.md (MODIFIED — code review learnings)
