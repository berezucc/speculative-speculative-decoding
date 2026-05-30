"""Verification-outcome prediction (§4.1 of Kumar et al. 2026).

Given K+1 draft logits and a fan-out budget {F_0, ..., F_K}, propose the most
likely (k_accepted, bonus_token) outcomes to populate the speculation cache.
"""
from __future__ import annotations

from typing import Callable

import torch

from .cache import Outcome

FanOutFn = Callable[[int, int], list[int]]


def uniform_fan_out(K: int, B: int) -> list[int]:
    """Distribute budget B as evenly as possible across K+1 positions."""
    base, rem = divmod(B, K + 1)
    return [base + (1 if i < rem else 0) for i in range(K + 1)]


def geometric_fan_out(K: int, B: int, acceptance_rate: float, r: float = 1.0) -> list[int]:
    """Capped geometric fan-out (Theorem 12, Kumar et al. 2026).

    Optimal F_k under a budget constraint when the rejection rate falls as 1/F^r:

        F_k = F_0 · a_p^(k/(1+r))                          for k < K
        F_K = F_0 · a_p^(K/(1+r)) · (1 - a_p)^(-1/(1+r))

    F_0 is chosen so sum(F_k) = B. Intuition: positions deeper in the speculation
    are less likely to be reached (need to accept k tokens), so fewer guesses.
    """
    if B <= 0:
        return [0] * (K + 1)
    if acceptance_rate <= 0:
        return [B] + [0] * K
    if acceptance_rate >= 1:
        return [0] * K + [B]

    exp = 1.0 / (1.0 + r)
    alpha = acceptance_rate ** exp
    beta = (1.0 - acceptance_rate) ** exp

    weights = [alpha ** k for k in range(K)] + [alpha ** K / beta]
    total = sum(weights)
    raw = [w * B / total for w in weights]

    F = [int(round(x)) for x in raw]
    diff = B - sum(F)
    if diff != 0:
        # Hand out (or take back) units to positions with largest fractional residue.
        residues = sorted(range(len(F)), key=lambda i: raw[i] - F[i], reverse=(diff > 0))
        step = 1 if diff > 0 else -1
        for i in residues[: abs(diff)]:
            F[i] += step
    return [max(0, f) for f in F]


def predict_outcomes(
    draft_logits: list[torch.Tensor],   # K+1 tensors of shape (V,)
    spec_tokens: list[int],             # K tokens already sent for verification
    fan_out: list[int],                 # K+1 entries
) -> list[Outcome]:
    """Pick the top-F_k draft tokens at each position k as bonus-token candidates.

    At positions k < K we exclude `spec_tokens[k]` — by construction it can't be
    the bonus token at that position (if accepted, the bonus comes from a later
    position; if rejected, the residual has zero mass on it).
    """
    if len(draft_logits) != len(fan_out):
        raise ValueError(f"draft_logits ({len(draft_logits)}) != fan_out ({len(fan_out)})")
    K = len(spec_tokens)
    outcomes: list[Outcome] = []
    for k, (logits, F) in enumerate(zip(draft_logits, fan_out)):
        if F == 0:
            continue
        top = torch.topk(logits, k=min(F + 1, logits.shape[-1])).indices.tolist()
        excluded = spec_tokens[k] if k < K else None
        for tok in top:
            if tok == excluded:
                continue
            outcomes.append(Outcome(k_accepted=k, bonus=tok))
            if sum(1 for o in outcomes if o.k_accepted == k) >= F:
                break
    return outcomes
