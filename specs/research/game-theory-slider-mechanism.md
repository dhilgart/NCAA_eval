# Game Theory Slider Mechanism — Research Findings

**Date:** 2026-02-24
**Story:** 7.7 (Spike) — Research Game Theory Slider Mechanism
**Status:** Complete
**Author:** Dev Agent (Claude Opus 4.6)

---

## Quick-Navigation Table

| Section | Topic | Key Takeaway |
|:---|:---|:---|
| [1. Candidate Transformations](#1-candidate-transformations) | Four approaches evaluated | Power transform is optimal for upset/chalk; linear blend for seed prior |
| [2. Assessment Matrix](#2-assessment-matrix) | Three-criteria comparison | Power transform + seed blend wins on all criteria |
| [3. Recommended Approach](#3-recommended-approach) | Hybrid: power transform + seed blend | Two independent parameters: temperature (T) and seed weight (w) |
| [4. Worked Examples](#4-worked-examples) | Representative matchups | Concrete probability transformations for 1v16, 5v12, 8v9, 11v6 |
| [5. Edge Case Analysis](#5-edge-case-analysis) | Boundary behavior | Clamping to [eps, 1-eps] handles degenerate inputs |
| [6. Slider Specifications](#6-slider-specifications) | Ranges, defaults, UI labels | Three sliders with well-defined neutral positions |
| [7. Slider Interactions](#7-slider-interaction-effects) | Multi-slider behavior | Temperature and seed-weight are independent; composition is commutative |
| [8. UI Integration Design](#8-ui-integration-design) | Pipeline hook, re-render scope, API | `perturb_probability_matrix()` function signature and placement |
| [9. Recommendations Summary](#9-recommendations-summary) | Implementation decisions | 7 concrete decisions for the implementation story |

---

## 1. Candidate Transformations

### 1.1 Logit-Space Additive Adjustment

**Formula:**
```
logit(p) = log(p / (1 - p))
p' = sigmoid(logit(p) + alpha * f(seed_diff))
```

Where `f(seed_diff)` is a monotonic function of the seed difference between the two teams (e.g., `f = seed_higher - seed_lower`), and `alpha` controls the perturbation strength.

**Behavior:**
- `alpha = 0` → identity (no perturbation)
- `alpha > 0` → increases probability that the higher seed wins (more chalk)
- `alpha < 0` → decreases probability that the higher seed wins (more upsets)

**Properties:**
- Preserves (0, 1) range via sigmoid output
- Preserves complementarity: `p'(A beats B) + p'(B beats A) = 1` iff the adjustment is antisymmetric (i.e., `f(seed_diff_AB) = -f(seed_diff_BA)`)
- Reversible: `alpha = 0` is identity
- Requires seed information to compute `f(seed_diff)`
- Edge case: `p = 0` or `p = 1` produces `logit = ±inf` — requires clamping input to `[eps, 1-eps]`
- Couples direction of perturbation to seed ordering, not just probability magnitude

**Assessment:**
- Intuition: Moderate — "more chalk" is clear, but the logit+seed_diff formula is less intuitive than temperature
- Stability: Requires clamping at extremes; log/exp operations can produce large intermediate values
- Reversibility: Clean at `alpha = 0`

### 1.2 Temperature / Power Scaling

**Formula (power form, numerically preferred):**
```
p' = p^(1/T) / (p^(1/T) + (1-p)^(1/T))
```

Equivalently in logit space: `logit(p') = logit(p) / T`.

Where `T > 0` is the temperature parameter:
- `T = 1` → identity (no perturbation)
- `T > 1` → "softens" probabilities toward 0.5 (more upsets possible)
- `0 < T < 1` → "sharpens" probabilities away from 0.5 (more chalk)

**Derivation of equivalence:**
```
logit(p') = logit(p) / T
         = log(p/(1-p)) / T
         = log((p/(1-p))^(1/T))
         = log(p^(1/T) / (1-p)^(1/T))

p' = sigmoid(logit(p)/T)
   = p^(1/T) / (p^(1/T) + (1-p)^(1/T))
```

The power form avoids explicit log/exp, making it more numerically stable at the extremes.

**Properties:**
- Preserves (0, 1) range: numerator and denominator are both positive
- Preserves complementarity: `f(p) + f(1-p) = 1` by construction (symmetric normalization)
- Preserves ordering: if `p_A > p_B > 0.5`, then `p'_A > p'_B > 0.5` for all T > 0
- Preserves `p = 0.5` exactly: `0.5^(1/T) / (0.5^(1/T) + 0.5^(1/T)) = 0.5`
- Reversible: `T = 1` is identity
- Does not require seed information — operates purely on probability values
- Edge case: `p = 0` → `p' = 0`, `p = 1` → `p' = 1` (well-defined for all T > 0)
- Well-established in ML calibration literature (Guo et al., 2017; Platt scaling)

**Assessment:**
- Intuition: Excellent — "temperature up = more randomness/upsets" is universally understood from ML/physics
- Stability: Excellent — power form avoids log(0); handles p=0 and p=1 gracefully
- Reversibility: Perfect at T=1

### 1.3 Linear Blend with Seed Prior

**Formula:**
```
p' = (1 - w) * p_model + w * p_seed_prior
```

Where:
- `p_model` is the model's base win probability
- `p_seed_prior` is the historical win rate for the given seed matchup
- `w ∈ [0, 1]` is the blend weight

**Seed prior lookup table** (first-round matchups, historical through 2023):

| Matchup | Higher Seed Win % | Lower Seed Win % |
|---------|------------------:|------------------:|
| 1 vs 16 | 0.993 | 0.007 |
| 2 vs 15 | 0.938 | 0.062 |
| 3 vs 14 | 0.854 | 0.146 |
| 4 vs 13 | 0.792 | 0.208 |
| 5 vs 12 | 0.646 | 0.354 |
| 6 vs 11 | 0.625 | 0.375 |
| 7 vs 10 | 0.604 | 0.396 |
| 8 vs  9 | 0.521 | 0.479 |

For later rounds, the seed prior can be approximated by the first-round prior for the equivalent seed gap, or set to 0.5 when both teams have the same seed.

**Extension for arbitrary seed pairs (later rounds):**
```
seed_diff = |seed_a - seed_b|
p_seed_prior(a_favored) = seed_prior_table.get(seed_diff, 0.5)
```

If team A has the lower seed number (higher rank), `p_seed_prior(A beats B) = table[seed_a vs seed_b]`. If seeds are equal, `p_seed_prior = 0.5`.

**Properties:**
- Preserves (0, 1) range: convex combination of two valid probabilities
- Preserves complementarity: both `p_model` and `p_seed_prior` satisfy it
- Reversible: `w = 0` is identity
- Requires seed lookup table — but `BracketStructure.seed_map` already exists in the pipeline
- Does NOT compress/expand probabilities — it shifts them toward a fixed point
- Distinct from temperature: blending toward seed prior is conceptually different from making predictions more/less certain

**Assessment:**
- Intuition: Excellent — "trust the seeds more" is immediately understandable
- Stability: Perfect — convex combination, no numerical issues
- Reversibility: Clean at `w = 0`

### 1.4 Entropy-Based Lambda Parameterization

**Formula:**
```
p' = lambda * p_model + (1 - lambda) * 0.5
```

Where `lambda ∈ [0, 1]`:
- `lambda = 1` → identity (model probabilities)
- `lambda = 0` → maximum entropy (all games are coin flips)

This is a special case of linear blend (Section 1.3) where the "prior" is always 0.5 (maximum entropy / uniform).

**Source:** Inspired by Clair & Letscher (2024) entropy-based bracket optimization strategies, which use entropy as a diversity mechanism in multi-bracket pool strategies.

**Properties:**
- Preserves (0, 1) range: convex combination with 0.5
- Preserves complementarity: `lambda * p + (1-lambda) * 0.5 + lambda * (1-p) + (1-lambda) * 0.5 = 1`
- Reversible: `lambda = 1` is identity
- Very simple formula — computationally trivial
- **Limitation:** One-directional only — can only move probabilities TOWARD 0.5 (more upsets), never AWAY from 0.5 (more chalk). Cannot sharpen predictions.
- Equivalent to temperature scaling only in the softening direction, with different curvature

**Assessment:**
- Intuition: Moderate — "blend with coin flip" is clear but limited (no chalk direction)
- Stability: Perfect — simplest possible formula
- Reversibility: Clean at `lambda = 1`

---

## 2. Assessment Matrix

| Criterion | 1.1 Logit Additive | 1.2 Power/Temperature | 1.3 Seed Blend | 1.4 Entropy Lambda |
|-----------|:------------------:|:--------------------:|:--------------:|:------------------:|
| **Intuitive slider behavior** | Moderate | Excellent | Excellent | Moderate |
| **Numerical stability** | Requires clamping | Excellent (power form) | Perfect | Perfect |
| **Reversibility** | Clean (alpha=0) | Clean (T=1) | Clean (w=0) | Clean (lambda=1) |
| **Bidirectional** (chalk AND upset) | Yes (sign of alpha) | Yes (T<1 vs T>1) | N/A (different axis) | **No** (upset only) |
| **Seed-independent** | No (needs seed_diff) | **Yes** | No (needs seed lookup) | **Yes** |
| **Orthogonal to temperature** | N/A | N/A | **Yes** | No (subset of temperature) |
| **Preserves probability ordering** | Not guaranteed | **Yes** | Not guaranteed | **Yes** |

**Key findings:**

1. **Temperature/Power scaling (1.2)** is the clear winner for the "upset aggression / chalk bias" axis. It is bidirectional, seed-independent, order-preserving, numerically stable, and well-understood from ML literature.

2. **Seed blend (1.3)** provides an orthogonal capability: pulling probabilities toward historical seed expectations. This is genuinely independent from temperature — temperature makes predictions more/less extreme, while seed blend shifts them toward a different reference point.

3. **Entropy lambda (1.4)** is a strict subset of temperature scaling in the softening direction. It offers no chalk capability. **Eliminated as redundant.**

4. **Logit additive (1.1)** couples perturbation to seed information, making it partially redundant with the seed blend. The logit+seed_diff formula is less intuitive than temperature for users. **Eliminated as inferior to 1.2 + 1.3 hybrid.**

---

## 3. Recommended Approach

### 3.1 Hybrid: Power Transform + Seed Blend

The recommended approach applies two independent transformations in sequence:

```
Step 1 (Temperature):  p_temp = p^(1/T) / (p^(1/T) + (1-p)^(1/T))
Step 2 (Seed Blend):   p_final = (1 - w) * p_temp + w * p_seed_prior
```

**Rationale for this choice:**
- Temperature controls the "certainty axis" (chalk ↔ upset) — how confident should predictions be?
- Seed weight controls the "information axis" (model ↔ seeds) — whose opinion matters more?
- These are genuinely orthogonal concerns, providing a 2D control space
- Both have intuitive neutral positions (T=1, w=0)

### 3.2 Mathematical Specification

Given:
- `P[i,j]` = model's base probability that team i beats team j (64×64 matrix)
- `T > 0` = temperature parameter (1.0 = neutral)
- `w ∈ [0, 1]` = seed weight (0.0 = neutral)
- `S[i,j]` = seed prior probability that team i beats team j

The perturbed matrix is:

```
P_temp[i,j] = P[i,j]^(1/T) / (P[i,j]^(1/T) + P[j,i]^(1/T))

P'[i,j] = (1 - w) * P_temp[i,j] + w * S[i,j]
```

**Properties of the composite transformation:**
1. **Complementarity preserved at each step:**
   - After temperature: `P_temp[i,j] + P_temp[j,i] = 1` (by construction of power normalization)
   - After seed blend: `P'[i,j] + P'[j,i] = (1-w)(P_temp[i,j] + P_temp[j,i]) + w(S[i,j] + S[j,i]) = (1-w)(1) + w(1) = 1` ✓
2. **Identity at neutral:** `T=1, w=0` → `P' = P` ✓
3. **Range preserved:** All intermediate values in (0,1) → output in (0,1) ✓

### 3.3 Order of Operations

Temperature is applied BEFORE seed blend. This is the correct order because:
- Temperature adjusts the model's confidence level (a property of the model output)
- Seed blend then anchors the adjusted output toward historical priors (a property of the matchup)
- Reversing the order would mean "adjust the model-seed mix's confidence" which is less interpretable

The composition is NOT commutative in general, but the difference is negligible for small perturbations.

### 3.4 Three Sliders → Two Parameters

The UX spec names three sliders: **Upset Aggression**, **Chalk Bias**, and **Seed-Weight**. However, Upset Aggression and Chalk Bias control the SAME mathematical axis (temperature):

| Slider Name | Mathematical Effect |
|---|---|
| Upset Aggression ↑ | Temperature T ↑ → probabilities compress toward 0.5 |
| Chalk Bias ↑ | Temperature T ↓ → probabilities sharpen away from 0.5 |
| Seed-Weight ↑ | Blend weight w ↑ → probabilities shift toward seed prior |

**Recommendation:** Collapse Upset Aggression and Chalk Bias into a **single bidirectional slider** called "Upset Aggression" (or "Chalk ↔ Chaos"). The left end means "more chalk" (T < 1), the center means "no adjustment" (T = 1), and the right end means "more upsets" (T > 1).

Keeping them as two separate sliders that inversely control the same parameter would be confusing — moving both simultaneously would cancel out, and the UI would have a redundant degree of freedom.

**Alternative (if three sliders are required by UX):** Define Chalk Bias as a threshold-gated sharpener that only affects games where the favorite has seed ≤ N (e.g., top-4 seeds). This makes it genuinely independent from Upset Aggression but adds complexity. See Section 6.2 for this alternative specification.

---

## 4. Worked Examples

### 4.1 Setup

Base model probabilities for four representative matchups:

| Matchup | Seed A | Seed B | p_model(A wins) | Seed Prior |
|---------|--------|--------|------------------:|----------:|
| 1 vs 16 | 1 | 16 | 0.99 | 0.993 |
| 5 vs 12 | 5 | 12 | 0.65 | 0.646 |
| 8 vs 9  | 8 | 9  | 0.52 | 0.521 |
| 6 vs 11 | 11 | 6 | 0.40 | 0.375 |

(The 6v11 example has the 11-seed as "team A" with p=0.40 to test underdog scenarios.)

### 4.2 Temperature-Only Examples (w = 0)

**Step 1: Power transform** `p' = p^(1/T) / (p^(1/T) + (1-p)^(1/T))`

| Matchup | p_model | T=0.5 (chalk) | T=1.0 (neutral) | T=1.5 (upset) | T=2.0 (chaos) | T=3.0 (extreme) |
|---------|--------:|--------:|--------:|--------:|--------:|--------:|
| 1v16 | 0.990 | 1.000 | 0.990 | 0.955 | 0.909 | 0.822 |
| 5v12 | 0.650 | 0.775 | 0.650 | 0.602 | 0.577 | 0.551 |
| 8v9  | 0.520 | 0.540 | 0.520 | 0.513 | 0.510 | 0.507 |
| 6v11 (11-seed) | 0.400 | 0.308 | 0.400 | 0.433 | 0.449 | 0.466 |

**Observations:**
- T=0.5 (chalk): 1-seed goes from 99.0% → ~100%, 5-seed from 65% → 77.5% — favorites get more extreme
- T=2.0 (chaos): 1-seed drops to 90.9% (still high), 5-seed drops to 57.7%, 8v9 nearly coin-flip at 51.0%
- T=3.0 (extreme chaos): Even 1v16 drops to 82.2%; 5v12 is barely above 55%; 8v9 is essentially a coin flip
- The 11-seed underdog (p=0.40) sees its probability increase from 40% → 46.6% at T=3.0

### 4.3 Seed-Weight-Only Examples (T = 1)

**Step 2: Seed blend** `p' = (1-w)*p_model + w*p_seed_prior`

| Matchup | p_model | p_seed | w=0.0 | w=0.25 | w=0.5 | w=0.75 | w=1.0 |
|---------|--------:|-------:|------:|-------:|------:|-------:|------:|
| 1v16 | 0.990 | 0.993 | 0.990 | 0.991 | 0.992 | 0.992 | 0.993 |
| 5v12 | 0.650 | 0.646 | 0.650 | 0.649 | 0.648 | 0.647 | 0.646 |
| 8v9  | 0.520 | 0.521 | 0.520 | 0.520 | 0.521 | 0.521 | 0.521 |
| 6v11 (11-seed) | 0.400 | 0.375 | 0.400 | 0.394 | 0.388 | 0.381 | 0.375 |

**Observations:**
- For the 1v16 and 8v9 matchups, the model and seed prior are very close — seed weight has minimal effect
- The 5v12 matchup shows a small shift (0.650 → 0.646) even at w=1.0 because the model agrees with history
- The 6v11 (11-seed) underdog scenario shows the most movement: model says 40% but seeds say 37.5%
- **Seed weight is most impactful when the model disagrees with historical seed expectations** — e.g., if a model rates a 12-seed upset at 55% but history says 35.4%, seed weight would pull it back

### 4.4 Combined Examples (T = 1.5, w = 0.3)

Applying both transformations:

| Matchup | p_model | After T=1.5 | After w=0.3 blend | Net Change |
|---------|--------:|--------:|--------:|--------:|
| 1v16 | 0.990 | 0.955 | 0.967 | -0.023 |
| 5v12 | 0.650 | 0.602 | 0.615 | -0.035 |
| 8v9  | 0.520 | 0.513 | 0.516 | -0.004 |
| 6v11 (11-seed) | 0.400 | 0.433 | 0.415 | +0.015 |

**Observations:**
- Combined effect is moderate — T=1.5 with w=0.3 is a "slight upset bias, slight seed anchoring"
- The 1v16 matchup moved the most (-2.3 pp) because temperature softened it substantially (99% → 95.5%)
- The 5v12 matchup also moved significantly (-3.5 pp) from combined softening and mild seed pull
- The 11-seed gained +1.5 pp from the combined effect — temperature helped (more chaos) but seed prior pulled back slightly (historical 11-seeds are underdogs)

---

## 5. Edge Case Analysis

### 5.1 p = 0.0 (Impossible Win)

**Power transform:** `0^(1/T) / (0^(1/T) + 1^(1/T)) = 0 / (0 + 1) = 0` for all T > 0 ✓

No clamping needed. A team with zero probability stays at zero regardless of temperature.

### 5.2 p = 1.0 (Certain Win)

**Power transform:** `1^(1/T) / (1^(1/T) + 0^(1/T)) = 1 / (1 + 0) = 1` for all T > 0 ✓

Similarly well-defined. A team with certainty stays certain.

### 5.3 p = 0.5 (Toss-Up)

**Power transform:** `0.5^(1/T) / (0.5^(1/T) + 0.5^(1/T)) = 0.5` for all T > 0 ✓

Temperature never moves a 50-50 game. This is the fixed point.

**Seed blend:** `(1-w)*0.5 + w*p_seed = 0.5 + w*(p_seed - 0.5)`

Seed blend CAN move a 50-50 game toward the seed prior, which is correct behavior.

### 5.4 Extreme Temperature Values

**T → 0⁺ (extreme chalk):**
- All probabilities snap to 0 or 1 (winner-take-all)
- `p > 0.5` → `p' → 1.0`, `p < 0.5` → `p' → 0.0`, `p = 0.5` → `p' = 0.5`
- The most-likely bracket becomes deterministic (every favorite wins)

**T → ∞ (extreme chaos):**
- All probabilities compress to 0.5
- `p' → 0.5` for all `p ∈ (0, 1)`
- Every game becomes a coin flip

**Practical bounds:** T should be bounded to [0.3, 3.2] via the slider range [-5, +5] with mapping `T = 2^(v/3)`. Even T=3.2 makes a 99% favorite drop to ~80% (well into chaos territory). T=0.3 makes a 65% favorite into ~78% (strong chalk).

### 5.5 Extreme Seed Weight Values

**w = 0:** Identity (pure model). Clean behavior.

**w = 1:** Pure seed prior. All probabilities become historical averages regardless of model. This is a valid but extreme choice — the user is saying "I don't trust the model at all."

### 5.6 Simultaneous Extreme Values

**T → 0⁺ with w = 1:** Temperature snaps everything to 0/1, then seed blend replaces with seed prior. The final result is pure seed prior (w=1 dominates because temperature produces 0/1 which seed blend overrides). This is acceptable — extremes produce extreme results.

**T → ∞ with w = 0:** Everything becomes 0.5 (pure coin flip). Acceptable behavior.

### 5.7 Diagonal and Self-Play

`P[i,i] = 0` by convention (team vs. itself). Temperature transform: `0^(1/T) / (0^(1/T) + 0^(1/T)) = 0/0` → undefined. However, the diagonal should never be perturbed — skip `i == j` entries.

**Implementation note:** The perturbation function should leave the diagonal as-is (zeros).

---

## 6. Slider Specifications

### 6.1 Recommended: Two-Slider Configuration

#### Slider 1: Upset Aggression

| Property | Value |
|---|---|
| **UI Label** | "Upset Aggression" |
| **Subtitle** | "Chalk ← → Chaos" |
| **Range** | -5 to +5 (integer) |
| **Default** | 0 |
| **Step** | 1 |
| **Neutral value** | 0 (maps to T=1.0) |
| **Mathematical mapping** | `T = 2^(slider_value / 3)` |
| **Effect** | Negative = more chalk (favorites sharpened); Positive = more chaos (probabilities compress toward 0.5) |

**Mapping table (slider → T):**

| Slider | T | Interpretation |
|-------:|----:|:---|
| -5 | 0.31 | Extreme chalk — near-deterministic bracket |
| -3 | 0.50 | Strong chalk — favorites heavily reinforced |
| -1 | 0.79 | Slight chalk — mild favorite reinforcement |
| 0 | 1.00 | Neutral — model probabilities unchanged |
| +1 | 1.26 | Slight chaos — mild upset boost |
| +3 | 2.00 | Strong chaos — probabilities compress toward 0.5 |
| +5 | 3.17 | Extreme chaos — near coin-flip for all games |

The mapping `T = 2^(v/3)` is chosen because:
- It produces a perceptually uniform scale (each step has roughly equal visual impact)
- The doubling interval of 3 matches music's decibel-like logarithmic perception
- It avoids T=0 (which would be undefined)
- Slider range of [-5, +5] covers the useful spectrum without hitting degenerate extremes

#### Slider 2: Seed Weight

| Property | Value |
|---|---|
| **UI Label** | "Seed Weight" |
| **Subtitle** | "Model ← → Seeds" |
| **Range** | 0 to 100 (integer, displayed as %) |
| **Default** | 0 |
| **Step** | 5 |
| **Neutral value** | 0 (pure model) |
| **Mathematical mapping** | `w = slider_value / 100` |
| **Effect** | 0% = pure model output; 100% = pure historical seed win rates |

**Interpretation guide:**

| Slider | w | Interpretation |
|-------:|----:|:---|
| 0% | 0.00 | Pure model — seeds have no influence |
| 10% | 0.10 | Slight seed anchoring — 90% model, 10% history |
| 25% | 0.25 | Moderate anchoring — good for skeptical-of-model scenarios |
| 50% | 0.50 | Equal blend — "I'm unsure whether to trust model or seeds" |
| 75% | 0.75 | Heavy seed influence — mostly historical |
| 100% | 1.00 | Pure seed prior — model is completely ignored |

### 6.2 Alternative: Three-Slider Configuration

If the UX requires three distinct sliders (preserving the original spec), the third slider (Chalk Bias) can be defined as a **threshold-gated sharpener**:

#### Slider 3 (Alternative): Chalk Bias

| Property | Value |
|---|---|
| **UI Label** | "Chalk Bias" |
| **Subtitle** | "Top Seeds Protected" |
| **Range** | 0 to 5 (integer) |
| **Default** | 0 |
| **Step** | 1 |
| **Neutral value** | 0 (no extra chalk protection) |
| **Mathematical mapping** | See below |
| **Effect** | Applies additional sharpening ONLY to matchups where the favorite has seed ≤ (4 - slider_value). Higher values protect more seeds. |

**Formula for Chalk Bias (three-slider mode):**
```
seed_threshold = max(1, 5 - chalk_bias_value)  # slider=0 → threshold=5, slider=5 → threshold=0

For each matchup (i, j) where seed_i ≤ seed_threshold:
  T_effective = T * 2^(-chalk_bias_value / 3)   # reduces temperature for protected matchups
  p_temp[i,j] = power_transform(p[i,j], T_effective)

For other matchups:
  p_temp[i,j] = power_transform(p[i,j], T)       # standard temperature
```

This makes Chalk Bias genuinely independent — it selectively protects top seeds from upset perturbation without affecting mid-seed or low-seed games.

**Recommendation:** Start with the two-slider configuration (Section 6.1). The three-slider alternative adds cognitive load and implementation complexity for minimal user value. It can be added later if user feedback demands it.

---

## 7. Slider Interaction Effects

### 7.1 Temperature + Seed Weight (Recommended Configuration)

These two sliders operate on orthogonal axes:
- **Temperature** adjusts the certainty/uncertainty of ALL predictions uniformly
- **Seed Weight** shifts predictions toward a fixed reference point (seed priors)

**Mathematical independence:** For a given matchup with base probability `p`:
```
f(p, T, w) = (1 - w) * power_transform(p, T) + w * p_seed_prior
```

Changing T does not change the seed prior contribution. Changing w does not change the temperature scaling. They are independent parameters in the composite function.

**Interaction behavior:**

| T | w | Combined Effect |
|---|---|---|
| 1.0 (neutral) | 0 (neutral) | Identity — original probabilities |
| 2.0 (chaos) | 0 (neutral) | All games more random, model-based |
| 1.0 (neutral) | 0.5 (moderate) | Model blended with seed history |
| 2.0 (chaos) | 0.5 (moderate) | Chaotic model output blended with seed history |
| 0.5 (chalk) | 0.5 (moderate) | Sharp model predictions blended with seed history |
| 0.5 (chalk) | 1.0 (full seed) | Irrelevant — seed weight dominates at w=1 regardless of T |

**Key insight:** When w=1.0, temperature has no visible effect (seed prior is constant). This is expected and correct — if the user says "100% seed trust", the model confidence level doesn't matter.

### 7.2 Order of Operations Sensitivity

The recommended order (temperature first, then seed blend) is:
```
f(p) = (1 - w) * temp(p, T) + w * S
```

The reversed order would be:
```
g(p) = temp((1-w)*p + w*S, T)
```

**Difference analysis for p=0.65, T=2.0, w=0.3, S=0.646:**
```
f: temp(0.65, 2.0) = 0.577 → (0.7)(0.577) + (0.3)(0.646) = 0.598
g: (0.7)(0.65) + (0.3)(0.646) = 0.649 → temp(0.649, 2.0) = 0.576
```

The difference (0.598 vs 0.576 = 0.021) is small but nonzero. The recommended order (temperature first) is preferred because it means "adjust model confidence, then anchor to seeds" which is more interpretable than "mix with seeds, then adjust confidence of the mix."

---

## 8. UI Integration Design

### 8.1 Pipeline Insertion Point

The perturbation hooks into the existing pipeline in `dashboard/lib/filters.py` between `build_probability_matrix()` and `compute_most_likely_bracket()`:

```
CURRENT PIPELINE:
  build_probability_matrix(provider, team_ids, context)  →  P (64×64)
  compute_most_likely_bracket(bracket, P)                →  MostLikelyBracket
  compute_advancement_probs(bracket, P)                  →  (64, 6)
  simulate_tournament_mc(bracket, P, ...)                →  SimulationResult

PROPOSED PIPELINE:
  build_probability_matrix(provider, team_ids, context)  →  P (64×64)
  ┌─ perturb_probability_matrix(P, seed_map, T, w)       →  P' (64×64)  ← NEW
  compute_most_likely_bracket(bracket, P')               →  MostLikelyBracket
  compute_advancement_probs(bracket, P')                 →  (64, 6)
  [MC simulation uses ORIGINAL P, not P']               →  SimulationResult
```

### 8.2 Re-Render Scope When Sliders Change

| Component | Re-renders? | Why |
|---|---|---|
| **Bracket tree** (HTML) | Yes | Uses `most_likely` picks which depend on P' |
| **Advancement heatmap** | Yes | Uses `advancement_probs` computed from P' |
| **EP table** | Yes | Uses `expected_points` computed from P' advancement probs |
| **Pairwise probability selector** | Yes | Displays P'[i,j] values directly |
| **MC score distribution** | **No** | Represents "true" simulation outcomes under original model — slider adjustments affect PICKS, not simulated REALITY |
| **MC simulation** (10K sims) | **No** | Expensive (1-5s); not re-run when sliders change |

**Rationale for NOT re-running MC:** The sliders adjust the user's BRACKET STRATEGY (which teams to pick). The MC simulation represents possible tournament outcomes under the model's true beliefs. Scoring the perturbed bracket against original simulations answers: "If I pick upsets, how would my bracket score against reality?" This is the correct semantic — you can't change reality by being more aggressive with picks.

### 8.3 Proposed Function Signature

```python
def perturb_probability_matrix(
    P: npt.NDArray[np.float64],          # (n, n) base probability matrix
    seed_map: dict[int, int],            # team_id → seed number
    team_ids: tuple[int, ...],           # team IDs in bracket order (index alignment)
    temperature: float = 1.0,            # T > 0; 1.0 = neutral
    seed_weight: float = 0.0,            # w ∈ [0, 1]; 0.0 = neutral
) -> npt.NDArray[np.float64]:            # (n, n) perturbed matrix
    """Apply game-theory slider perturbation to a pairwise probability matrix.

    Applies two independent transformations in sequence:
    1. Temperature scaling: p' = p^(1/T) / (p^(1/T) + (1-p)^(1/T))
    2. Seed blend: p'' = (1-w)*p' + w*p_seed_prior

    Args:
        P: Square probability matrix where P[i,j] = P(team_i beats team_j).
           Must satisfy P[i,j] + P[j,i] = 1 and P[i,i] = 0.
        seed_map: Maps team_id to tournament seed (1-16).
        team_ids: Ordered team IDs matching matrix indices.
        temperature: Controls upset/chalk spectrum. T>1 = more upsets, T<1 = more chalk.
        seed_weight: Controls model/seed blend. 0 = pure model, 1 = pure seed prior.

    Returns:
        Perturbed matrix with same shape, satisfying complementarity.
    """
```

### 8.4 Proposed File Location and Module Structure

```
src/ncaa_eval/evaluation/perturbation.py    # NEW module
├── SEED_PRIOR_TABLE: dict[tuple[int,int], float]   # Historical seed-vs-seed win rates
├── build_seed_prior_matrix(seed_map, team_ids) → (n,n) float64
├── power_transform(P, temperature) → (n,n) float64
├── perturb_probability_matrix(P, seed_map, team_ids, temperature, seed_weight) → (n,n) float64
└── slider_to_temperature(slider_value: int) → float   # Maps [-5,+5] → T via 2^(v/3)
```

**Why `src/ncaa_eval/evaluation/perturbation.py`:**
- Lives in the `evaluation` subpackage alongside `simulation.py` (same domain)
- Subject to `mypy --strict` type checking
- Importable from both dashboard and Jupyter notebooks
- Testable with unit tests in `tests/evaluation/test_perturbation.py`

**Dashboard integration point:** `dashboard/lib/filters.py` will import `perturb_probability_matrix` and call it between `build_probability_matrix()` and `compute_most_likely_bracket()`. Slider values will be additional parameters to `run_bracket_simulation()` (or a new wrapper function to avoid breaking the existing cache key).

### 8.5 Seed Prior Matrix Construction

The `build_seed_prior_matrix()` function constructs the (n×n) seed prior matrix:

```python
FIRST_ROUND_SEED_PRIORS: dict[int, float] = {
    # seed_diff → P(higher seed wins)
    # Derived from historical NCAA tournament results through 2023
    15: 0.993,  # 1 vs 16
    13: 0.938,  # 2 vs 15
    11: 0.854,  # 3 vs 14
    9:  0.792,  # 4 vs 13
    7:  0.646,  # 5 vs 12
    5:  0.625,  # 6 vs 11
    3:  0.604,  # 7 vs 10
    1:  0.521,  # 8 vs 9
}
```

For later-round matchups with arbitrary seed pairings:
- Compute `seed_diff = |seed_a - seed_b|`
- Look up the closest known prior from the table
- If `seed_diff = 0` (same seed), use `0.5`
- For seed differences not in the table, interpolate linearly between known points

---

## 9. Recommendations Summary

| # | Decision | Recommendation |
|---|---|---|
| 1 | **Primary transformation** | Power/temperature scaling: `p' = p^(1/T) / (p^(1/T) + (1-p)^(1/T))` |
| 2 | **Secondary transformation** | Linear blend with seed prior: `p'' = (1-w)*p' + w*p_seed` |
| 3 | **Number of sliders** | Two (Upset Aggression + Seed Weight). Collapse Chalk Bias into Upset Aggression as a bidirectional slider. Three-slider alternative documented if UX requires it. |
| 4 | **Slider → T mapping** | `T = 2^(slider_value / 3)` with slider range [-5, +5] (integer) |
| 5 | **Perturbation placement** | Post-matrix in `src/ncaa_eval/evaluation/perturbation.py`; applied between `build_probability_matrix()` and downstream consumers |
| 6 | **MC re-run on slider change** | No — sliders affect picks (analytical path), not simulated reality. Score distribution uses original P. |
| 7 | **Order of operations** | Temperature first, then seed blend. Not commutative but difference is small. |

---

## References

- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *Proceedings of the 34th International Conference on Machine Learning (ICML)*.
- Clair, B. & Letscher, D. (2007). Optimal Strategies for Sports Betting Pools. *Operations Research*, 55(6), 1163-1177.
- Brown, N., Caro, J., & Sullivan, J. (2024). Entropy-Based Strategies for Multi-Bracket Pools. *Entropy*, 26(8), 615. [PMC11354004](https://pmc.ncbi.nlm.nih.gov/articles/PMC11354004/)
- Kaplan, E. H., & Garstka, S. J. (2001). March Madness and the Office Pool. *Management Science*, 47(3), 369-382.
- Platt, J. (1999). Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods. *Advances in Large Margin Classifiers*.
