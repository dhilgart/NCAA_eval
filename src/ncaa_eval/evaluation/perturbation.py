"""Game-theory slider perturbation for bracket probability matrices.

Applies two independent transformations to a pairwise probability matrix:

1. **Temperature (Upset Aggression):** Power scaling that compresses
   probabilities toward 0.5 (more upsets, T > 1) or sharpens them away
   from 0.5 (more chalk, T < 1).

2. **Seed Blend (Seed-Weight):** Linear interpolation between the
   (possibly temperature-adjusted) model output and historical
   first-round seed priors.

Both transformations preserve complementarity:
``P[i,j] + P[j,i] = 1`` for all ``(i, j)`` pairs.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# Historical first-round win rates by seed difference (higher seed wins).
# Derived from NCAA tournament results through 2023.
# Keys are odd seed differences from first-round matchups (1v16=15, ..., 8v9=1).
FIRST_ROUND_SEED_PRIORS: dict[int, float] = {
    15: 0.993,  # 1 vs 16
    13: 0.938,  # 2 vs 15
    11: 0.854,  # 3 vs 14
    9: 0.792,  # 4 vs 13
    7: 0.646,  # 5 vs 12
    5: 0.625,  # 6 vs 11
    3: 0.604,  # 7 vs 10
    1: 0.521,  # 8 vs 9
}

# Sorted keys for interpolation lookups.
_SORTED_DIFFS: list[int] = sorted(FIRST_ROUND_SEED_PRIORS)
_MAX_DIFF: int = _SORTED_DIFFS[-1]


def _lookup_seed_prior(seed_diff: int) -> float:
    """Return P(higher seed wins) for a given absolute seed difference.

    Uses direct lookup for tabulated values (odd differences 1–15),
    linear interpolation for even differences, 0.5 for same-seed
    matchups, and clamping for seed_diff > 15.
    """
    if seed_diff == 0:
        return 0.5
    if seed_diff >= _MAX_DIFF:
        return FIRST_ROUND_SEED_PRIORS[_MAX_DIFF]
    if seed_diff in FIRST_ROUND_SEED_PRIORS:
        return FIRST_ROUND_SEED_PRIORS[seed_diff]
    # Linear interpolation between adjacent tabulated values.
    d_low = max(d for d in _SORTED_DIFFS if d < seed_diff)
    d_high = min(d for d in _SORTED_DIFFS if d > seed_diff)
    p_low = FIRST_ROUND_SEED_PRIORS[d_low]
    p_high = FIRST_ROUND_SEED_PRIORS[d_high]
    frac = (seed_diff - d_low) / (d_high - d_low)
    return p_low + frac * (p_high - p_low)


def slider_to_temperature(slider_value: int) -> float:
    """Map an integer slider value to a temperature parameter.

    Args:
        slider_value: Integer in ``[-5, +5]``.

    Returns:
        ``T = 2^(slider_value / 3)``.  T=1.0 at slider_value=0 (neutral).

    Raises:
        ValueError: If *slider_value* is outside ``[-5, +5]``.
    """
    if not (-5 <= slider_value <= 5):
        msg = f"slider_value must be in [-5, +5], got {slider_value}"
        raise ValueError(msg)
    return float(2.0 ** (slider_value / 3.0))


def power_transform(
    P: npt.NDArray[np.float64],
    temperature: float,
) -> npt.NDArray[np.float64]:
    """Apply power/temperature scaling to a probability matrix.

    Computes ``p' = p^(1/T) / (p^(1/T) + (1-p)^(1/T))`` element-wise.

    Properties:
      - T=1.0: identity (p' = p).
      - T>1: compresses probabilities toward 0.5 (more upsets).
      - T<1: sharpens probabilities away from 0.5 (more chalk).
      - Preserves p=0, p=1, p=0.5 as fixed points.
      - Preserves diagonal zeros.
      - Preserves complementarity: ``p'[i,j] + p'[j,i] = 1``.

    Args:
        P: Square probability matrix with diagonal zeros.
        temperature: Temperature T > 0.

    Returns:
        Transformed matrix with same shape and dtype.

    Raises:
        ValueError: If *temperature* is not positive.
    """
    if temperature <= 0:
        msg = f"temperature must be positive, got {temperature}"
        raise ValueError(msg)
    if temperature == 1.0:
        return P.copy()

    inv_t = 1.0 / temperature

    # Save diagonal — power transform on 0/0 is undefined.
    diag = np.diag(P).copy()

    p_pow = np.power(P, inv_t)
    q_pow = np.power(1.0 - P, inv_t)

    denom = p_pow + q_pow
    result: npt.NDArray[np.float64] = np.divide(p_pow, denom)

    # Restore diagonal zeros (diagonal entries have p=0, so p_pow=0 and
    # denom=1, giving result=0 — explicit restore keeps precision exact).
    np.fill_diagonal(result, diag)

    return result


def build_seed_prior_matrix(
    seed_map: dict[int, int],
    team_ids: Sequence[int],
) -> npt.NDArray[np.float64]:
    """Build an (n × n) seed prior probability matrix.

    ``S[i, j] = P(team_i beats team_j)`` based on historical seed win
    rates keyed by ``|seed_i − seed_j|``.

    Args:
        seed_map: Maps ``team_id → seed_number`` (1–16).
        team_ids: Ordered team IDs matching matrix indices.

    Returns:
        Float64 matrix of shape ``(n, n)`` with diagonal zeros and
        ``S[i,j] + S[j,i] = 1``.
    """
    seeds = np.array([seed_map.get(tid, 0) for tid in team_ids], dtype=np.int32)

    missing = [tid for tid in team_ids if tid not in seed_map]
    if missing:
        logger.warning(
            "build_seed_prior_matrix: %d team(s) missing from seed_map — "
            "treating as seed=0 (same-seed coin flip): %s",
            len(missing),
            missing[:10],
        )

    # Vectorized seed-diff matrix: (n, n) with abs(seed_i - seed_j).
    diff_matrix = np.abs(seeds[:, None] - seeds[None, :]).astype(np.int32)

    # Vectorize the scalar lookup over all pairwise seed differences.
    # np.vectorize applies _lookup_seed_prior element-wise on the diff matrix
    # to produce P(higher_seed wins) for each (i, j) pair.
    _vec_lookup = np.vectorize(_lookup_seed_prior)
    p_higher_wins: npt.NDArray[np.float64] = _vec_lookup(diff_matrix).astype(np.float64)

    # Build S[i, j] = P(team_i beats team_j):
    # - If seeds[i] < seeds[j]: team i is higher seed → S[i,j] = p_higher_wins[i,j]
    # - If seeds[i] > seeds[j]: team j is higher seed → S[i,j] = 1 - p_higher_wins[i,j]
    # - If seeds[i] == seeds[j]: coin flip → S[i,j] = 0.5 (already from _lookup at diff=0)
    higher_seed_mask = seeds[:, None] < seeds[None, :]  # True where i is higher seed
    lower_seed_mask = seeds[:, None] > seeds[None, :]  # True where j is higher seed

    S = np.where(
        higher_seed_mask, p_higher_wins, np.where(lower_seed_mask, 1.0 - p_higher_wins, p_higher_wins)
    )
    np.fill_diagonal(S, 0.0)
    return S.astype(np.float64)


def perturb_probability_matrix(
    P: npt.NDArray[np.float64],
    seed_map: dict[int, int],
    team_ids: Sequence[int],
    temperature: float = 1.0,
    seed_weight: float = 0.0,
) -> npt.NDArray[np.float64]:
    """Apply game-theory slider perturbation to a pairwise probability matrix.

    Applies two independent transformations in sequence:

    1. Temperature scaling: ``p' = p^(1/T) / (p^(1/T) + (1-p)^(1/T))``
    2. Seed blend: ``p'' = (1-w)*p' + w*p_seed_prior``

    Args:
        P: Square probability matrix where ``P[i,j]`` is the probability
           that team *i* beats team *j*.  Must satisfy
           ``P[i,j] + P[j,i] = 1`` and ``P[i,i] = 0``.
        seed_map: Maps ``team_id → seed_number`` (1–16).
        team_ids: Ordered team IDs matching matrix indices.
        temperature: Controls upset/chalk spectrum.  T > 1 = more upsets,
                     T < 1 = more chalk, T = 1 = neutral.
        seed_weight: Controls model/seed blend.  0 = pure model,
                     1 = pure seed prior.

    Returns:
        Perturbed matrix of same shape satisfying complementarity.

    Raises:
        ValueError: If *temperature* ≤ 0 or *seed_weight* not in ``[0, 1]``.
    """
    if temperature <= 0:
        msg = f"temperature must be positive, got {temperature}"
        raise ValueError(msg)
    if not (0.0 <= seed_weight <= 1.0):
        msg = f"seed_weight must be in [0, 1], got {seed_weight}"
        raise ValueError(msg)

    # Fast path: neutral sliders → no-op.
    if temperature == 1.0 and seed_weight == 0.0:
        return P.copy()

    # Step 1: Temperature transform.
    result = power_transform(P, temperature)

    # Step 2: Seed blend.
    if seed_weight > 0.0:
        S = build_seed_prior_matrix(seed_map, team_ids)
        result = (1.0 - seed_weight) * result + seed_weight * S

    return result
