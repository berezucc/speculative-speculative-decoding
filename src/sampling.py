"""Sampling primitives shared by SD and SSD."""
from __future__ import annotations

import math

import torch


def logits_to_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Convert logits → probability distribution. temperature=0 returns a one-hot argmax."""
    if temperature == 0.0:
        probs = torch.zeros_like(logits)
        if logits.dim() == 1:
            probs[torch.argmax(logits)] = 1.0
        else:
            idx = torch.argmax(logits, dim=-1)
            probs.scatter_(-1, idx.unsqueeze(-1), 1.0)
        return probs
    return torch.softmax(logits / temperature, dim=-1)


def sample(probs: torch.Tensor) -> torch.Tensor:
    """Sample one token id from a 1-D probability vector. Greedy if the dist is one-hot."""
    if probs.max().item() == 1.0:
        return torch.argmax(probs)
    return torch.multinomial(probs, num_samples=1).squeeze()


def acceptance_alpha(p_target: torch.Tensor, p_draft: torch.Tensor, token: int) -> float:
    """min(1, p_target(token) / p_draft(token))."""
    return min(1.0, (p_target[token] / (p_draft[token] + 1e-10)).item())


def saguaro_probs(logits: torch.Tensor, temperature: float, F: int, C: float) -> torch.Tensor:
    """Saguaro sampling scheme σ_F,C from Definition 14 of Kumar et al. 2026.

    Downweights the top-F draft tokens by factor C ∈ [0, 1] in exp-space, pushing
    residual mass onto those same tokens (which are the cache candidates).
    Falls back to vanilla sampling when F ≤ 0 or C == 1.
    """
    if F <= 0 or C >= 1.0:
        return logits_to_probs(logits, temperature)
    biased = logits.clone()
    top = torch.topk(biased, k=min(F, biased.shape[-1])).indices
    if C == 0.0:
        biased[top] = float("-inf")
    else:
        biased[top] = biased[top] + math.log(C)
    return logits_to_probs(biased, temperature)


def sample_residual(p_target: torch.Tensor, p_draft: torch.Tensor) -> int:
    """Sample from r ∝ max(p_target − p_draft, 0)."""
    residual = torch.clamp(p_target - p_draft, min=0.0)
    total = residual.sum()
    if total <= 0:
        return int(torch.argmax(p_target).item())
    residual = residual / total
    return int(torch.multinomial(residual, num_samples=1).item())
