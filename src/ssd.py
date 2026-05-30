"""Speculative Speculative Decoding (Kumar, Dao & May 2026), single-device study.

On real SSD hardware, the speculator runs on a separate device in parallel with
the verifier. Here both run sequentially on one device — output is identical and
cache-hit dynamics are preserved exactly, but wall-clock is slower.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch

from .cache import CachedSpeculation, Outcome, SpeculationCache
from .models import LM, truncate_kv
from .outcomes import predict_outcomes, uniform_fan_out
from .sampling import logits_to_probs, sample, sample_residual


@dataclass
class SSDStats:
    rounds: int = 0
    cache_hits: int = 0
    total_proposed: int = 0
    total_accepted: int = 0
    cache_sizes: list[int] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return self.cache_hits / self.rounds if self.rounds else 0.0

    @property
    def acceptance_rate(self) -> float:
        return self.total_accepted / self.total_proposed if self.total_proposed else 0.0


def _prefill(lm: LM, ids: torch.Tensor):
    if ids.shape[1] <= 1:
        return None
    return lm.forward(ids[:, :-1]).past_key_values


def _draft_K_from(draft: LM, kv_init, first_token: int, K: int, temperature: float):
    """K+1 draft passes from `kv_init`, starting by feeding `first_token`.

    If `kv_init` covers prefix P, returns:
        spec_tokens   (K,)        spec[k] sampled from logits at position P+first_token+spec[:k-1]
        probs         (K+1, V)    probs[k] = draft distribution at position P+first_token+spec[:k]
                                  (the (K+1)th row predicts the bonus token for full-accept)
        snapshots     (K+1)       snap[k] = deep-copied KV covering P + first_token + spec[:k]
    """
    snapshots = []
    spec_tokens: list[torch.Tensor] = []
    probs: list[torch.Tensor] = []

    kv = kv_init
    prev = torch.tensor([[first_token]], device=draft.device)
    for _ in range(K):
        out = draft.forward(prev, past_kv=kv)
        kv = out.past_key_values
        snapshots.append(copy.deepcopy(kv))
        p = logits_to_probs(out.logits[0, -1, :], temperature)
        t = sample(p)
        spec_tokens.append(t)
        probs.append(p)
        prev = t.view(1, 1)

    out = draft.forward(prev, past_kv=kv)
    snapshots.append(copy.deepcopy(out.past_key_values))
    probs.append(logits_to_probs(out.logits[0, -1, :], temperature))

    return torch.stack(spec_tokens), torch.stack(probs), snapshots


def _build_cache(
    draft: LM,
    outcomes: list[Outcome],
    snapshots: list,
    k: int,
    temperature: float,
) -> SpeculationCache:
    """For each predicted outcome, pre-draft K next-round spec tokens."""
    cache = SpeculationCache()
    for outcome in outcomes:
        toks, probs, snaps = _draft_K_from(
            draft, copy.deepcopy(snapshots[outcome.k_accepted]), outcome.bonus, k, temperature
        )
        cache.put(
            outcome,
            CachedSpeculation(tokens=toks.tolist(), draft_probs=probs, kv_after=snaps[-1]),
        )
        # Attach snapshots as an attribute for next-round reuse (kept off the dataclass
        # to avoid bloating the cache size if not used).
        cache.entries[outcome].snapshots = snaps  # type: ignore[attr-defined]
    return cache


def _verify(
    verifier: LM,
    last_token: int,
    spec_tokens: torch.Tensor,
    spec_probs: torch.Tensor,
    v_kv,
    temperature: float,
):
    """One verifier forward pass on [last_token, spec[0..K-1]]; accept/reject + sample bonus."""
    K = spec_tokens.shape[0]
    v_input = torch.cat(
        [torch.tensor([[last_token]], device=verifier.device), spec_tokens.view(1, -1)], dim=1
    )
    out = verifier.forward(v_input, past_kv=v_kv)
    v_kv = out.past_key_values
    v_probs = logits_to_probs(out.logits[0], temperature)  # (K+1, V)

    new_tokens: list[int] = []
    rejected = False
    for i in range(K):
        tok = int(spec_tokens[i].item())
        alpha = min(1.0, (v_probs[i, tok] / (spec_probs[i, tok] + 1e-10)).item())
        if torch.rand(1).item() < alpha:
            new_tokens.append(tok)
        else:
            new_tokens.append(sample_residual(v_probs[i], spec_probs[i]))
            rejected = True
            break

    accepted = len(new_tokens) - (1 if rejected else 0)
    if not rejected:
        new_tokens.append(int(sample(v_probs[K]).item()))

    outcome = Outcome(k_accepted=accepted, bonus=new_tokens[-1])
    return outcome, new_tokens, accepted, v_kv


def ssd_decode(
    draft: LM,
    verifier: LM,
    prompt: str,
    n_tokens: int,
    k: int = 4,
    budget: int = 8,
    temperature: float = 0.0,
) -> tuple[list[int], SSDStats]:
    """SSD with uniform fan-out and synchronous-draft fallback on cache miss."""
    ids = verifier.tokenizer.encode(prompt, return_tensors="pt").to(verifier.device)
    L0 = ids.shape[1]
    current = ids.clone()

    v_kv = _prefill(verifier, ids)
    d_kv = _prefill(draft, ids)
    stats = SSDStats()

    last = int(current[0, -1].item())
    spec_tokens, spec_probs, snapshots = _draft_K_from(draft, d_kv, last, k, temperature)

    while current.shape[1] - L0 < n_tokens:
        T = current.shape[1]

        outcome, new_tokens, accepted, v_kv = _verify(
            verifier, last, spec_tokens, spec_probs[:k], v_kv, temperature
        )

        fan_out = uniform_fan_out(k, budget)
        predicted = predict_outcomes(
            [spec_probs[i] for i in range(k + 1)], spec_tokens.tolist(), fan_out
        )
        cache = _build_cache(draft, predicted, snapshots, k, temperature)
        stats.cache_sizes.append(len(cache))

        if accepted < k:
            v_kv = truncate_kv(v_kv, T + accepted)
        current = torch.cat(
            [current, torch.tensor([new_tokens], dtype=current.dtype, device=current.device)],
            dim=1,
        )
        stats.rounds += 1
        stats.total_proposed += k
        stats.total_accepted += accepted

        if current.shape[1] - L0 >= n_tokens:
            break

        last = int(current[0, -1].item())
        if outcome in cache:
            stats.cache_hits += 1
            cached = cache.get(outcome)
            spec_tokens = torch.tensor(cached.tokens, device=draft.device)
            spec_probs = cached.draft_probs
            snapshots = cached.snapshots  # type: ignore[attr-defined]
        else:
            spec_tokens, spec_probs, snapshots = _draft_K_from(
                draft, snapshots[accepted], last, k, temperature
            )

    return current[0, L0 : L0 + n_tokens].tolist(), stats
