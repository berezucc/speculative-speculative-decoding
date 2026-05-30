from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LM:
    """Causal LM with its tokenizer pinned to a device."""

    name: str
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: torch.device

    @classmethod
    def load(cls, name: str, device: torch.device) -> "LM":
        tok = AutoTokenizer.from_pretrained(name)
        mdl = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
        return cls(name=name, model=mdl, tokenizer=tok, device=device)

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, past_kv=None):
        return self.model(input_ids, past_key_values=past_kv, use_cache=True)


def assert_shared_vocab(a: LM, b: LM) -> None:
    if a.tokenizer.get_vocab() != b.tokenizer.get_vocab():
        raise ValueError(
            f"speculative decoding requires shared vocab; got {a.name} vs {b.name}"
        )


def truncate_kv(past_kv, target_len: Optional[int]):
    """Crop a KV cache to `target_len` (works for both Cache objects and legacy tuples)."""
    if past_kv is None or target_len is None:
        return past_kv
    if hasattr(past_kv, "crop"):
        past_kv.crop(target_len)
        return past_kv
    out = []
    for k, v in past_kv:
        if k.shape[-2] <= target_len:
            out.append((k, v))
        else:
            out.append((k[..., :target_len, :], v[..., :target_len, :]))
    return tuple(out)
