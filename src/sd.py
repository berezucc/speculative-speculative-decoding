"""Ordinary speculative decoding (Leviathan et al. 2023)."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .models import LM, truncate_kv
from .sampling import logits_to_probs, sample, sample_residual


@dataclass
class SDStats:
    total_proposed: int = 0
    total_accepted: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.total_accepted / self.total_proposed if self.total_proposed else 0.0


def _prefill(lm: LM, ids: torch.Tensor):
    if ids.shape[1] <= 1:
        return None
    return lm.forward(ids[:, :-1]).past_key_values


def _draft_k(draft: LM, last_token: int, k: int, temperature: float, draft_kv):
    """Run draft autoregressively for K steps. Returns (tokens (K,), probs (K, V), updated kv)."""
    tokens, probs = [], []
    prev = torch.tensor([[last_token]], device=draft.device)
    for _ in range(k):
        out = draft.forward(prev, past_kv=draft_kv)
        draft_kv = out.past_key_values
        p = logits_to_probs(out.logits[0, -1, :], temperature)
        t = sample(p)
        probs.append(p)
        tokens.append(t)
        prev = t.view(1, 1)
    return torch.stack(tokens), torch.stack(probs), draft_kv


def speculative_decode(
    draft: LM,
    verifier: LM,
    prompt: str,
    n_tokens: int,
    k: int = 4,
    temperature: float = 0.0,
) -> tuple[list[int], SDStats]:
    """Generate `n_tokens` from the verifier distribution via speculative decoding.

    Invariant maintained across iterations: each model's KV cache covers all of `current_ids`
    except the trailing token, which is fed at the start of the next iter.
    """
    ids = verifier.tokenizer.encode(prompt, return_tensors="pt").to(verifier.device)
    L0 = ids.shape[1]
    current = ids.clone()

    v_kv = _prefill(verifier, ids)
    d_kv = _prefill(draft, ids)
    stats = SDStats()

    while current.shape[1] - L0 < n_tokens:
        T = current.shape[1]
        last = int(current[0, -1].item())

        # 1. Draft K tokens
        d_tokens, d_probs, d_kv = _draft_k(draft, last, k, temperature, d_kv)

        # 2. Verify in one forward pass on [last, t1, ..., tK]
        v_input = torch.cat([torch.tensor([[last]], device=verifier.device), d_tokens.view(1, -1)], dim=1)
        out_v = verifier.forward(v_input, past_kv=v_kv)
        v_kv = out_v.past_key_values
        v_probs = logits_to_probs(out_v.logits[0], temperature)  # (K+1, V)

        # 3. Accept/reject loop
        new_tokens: list[int] = []
        rejected = False
        for i in range(k):
            tok = int(d_tokens[i].item())
            alpha = min(1.0, (v_probs[i, tok] / (d_probs[i, tok] + 1e-10)).item())
            if torch.rand(1).item() < alpha:
                new_tokens.append(tok)
            else:
                new_tokens.append(sample_residual(v_probs[i], d_probs[i]))
                rejected = True
                break

        n_accepted = len(new_tokens) - (1 if rejected else 0)

        # 4. Cache repair + bonus token
        if rejected:
            target = T + n_accepted
            v_kv = truncate_kv(v_kv, target)
            d_kv = truncate_kv(d_kv, target)
        else:
            new_tokens.append(int(sample(v_probs[k]).item()))
            # draft kv lags verifier by one; feed tK to catch up
            tK = d_tokens[k - 1].view(1, 1)
            d_kv = draft.forward(tK, past_kv=d_kv).past_key_values

        current = torch.cat(
            [current, torch.tensor([new_tokens], dtype=current.dtype, device=current.device)],
            dim=1,
        )
        stats.total_proposed += k
        stats.total_accepted += n_accepted

    return current[0, L0 : L0 + n_tokens].tolist(), stats


def greedy_decode(lm: LM, prompt: str, n_tokens: int) -> list[int]:
    """Verifier-only greedy decoding — the reference distribution that SD must match at temp=0."""
    ids = lm.tokenizer.encode(prompt, return_tensors="pt").to(lm.device)
    L0 = ids.shape[1]
    kv = None
    for _ in range(n_tokens):
        feed = ids if kv is None else ids[:, -1:]
        out = lm.forward(feed, past_kv=kv)
        kv = out.past_key_values
        nxt = torch.argmax(out.logits[0, -1, :]).view(1, 1)
        ids = torch.cat([ids, nxt], dim=1)
    return ids[0, L0:].tolist()
