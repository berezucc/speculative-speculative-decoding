"""Verification-outcome prediction (§4.1 of Kumar et al. 2026).

Given K+1 draft logits and a fan-out budget {F_0, ..., F_K}, propose the most
likely (k_accepted, bonus_token) outcomes to populate the speculation cache.
"""
from __future__ import annotations

import torch

from .cache import Outcome


def uniform_fan_out(K: int, B: int) -> list[int]:
    """Distribute budget B as evenly as possible across K+1 positions."""
    base, rem = divmod(B, K + 1)
    return [base + (1 if i < rem else 0) for i in range(K + 1)]


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
