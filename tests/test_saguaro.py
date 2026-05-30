"""Saguaro sampling (§4.2) properties + lossless preservation."""
import pytest
import torch

from src.models import LM, assert_shared_vocab
from src.sampling import logits_to_probs, saguaro_probs
from src.sd import greedy_decode
from src.ssd import ssd_decode
from src.utils import get_device, set_seed


def test_saguaro_identity_when_C_is_one():
    logits = torch.tensor([5.0, 3.0, 4.0, 1.0, 2.0])
    expected = logits_to_probs(logits, temperature=0.7)
    got = saguaro_probs(logits, temperature=0.7, F=3, C=1.0)
    assert torch.allclose(expected, got)


def test_saguaro_identity_when_F_is_zero():
    logits = torch.tensor([5.0, 3.0, 4.0, 1.0, 2.0])
    expected = logits_to_probs(logits, temperature=0.7)
    got = saguaro_probs(logits, temperature=0.7, F=0, C=0.1)
    assert torch.allclose(expected, got)


def test_saguaro_downweights_top_F():
    # logits favor token 0 strongly. With C=0.1 on top-2, mass moves off tokens {0, 2}.
    logits = torch.tensor([5.0, 1.0, 4.0, 0.0, 0.5])
    p_orig = logits_to_probs(logits, temperature=1.0)
    p_sag = saguaro_probs(logits, temperature=1.0, F=2, C=0.1)
    # Top-2 (tokens 0 and 2) should lose probability mass; others gain.
    assert p_sag[0] < p_orig[0]
    assert p_sag[2] < p_orig[2]
    assert p_sag[1] > p_orig[1]


def test_saguaro_C_zero_zeros_out_top_F():
    logits = torch.tensor([5.0, 1.0, 4.0, 0.0, 0.5])
    p = saguaro_probs(logits, temperature=1.0, F=2, C=0.0)
    assert p[0].item() == 0.0  # was top-1
    assert p[2].item() == 0.0  # was top-2
    assert abs(p.sum().item() - 1.0) < 1e-5


@pytest.fixture(scope="module")
def models():
    device = get_device()
    draft = LM.load("distilgpt2", device)
    verifier = LM.load("distilgpt2", device)
    assert_shared_vocab(draft, verifier)
    return draft, verifier


@pytest.mark.parametrize("C", [0.5, 0.1])
def test_ssd_lossless_with_saguaro(models, C):
    """SSD must remain lossless at temp=0 regardless of Saguaro C."""
    draft, verifier = models
    set_seed()
    ref = greedy_decode(verifier, "The transformer architecture", 12)
    set_seed()
    spec, stats = ssd_decode(
        draft, verifier, "The transformer architecture", 12,
        k=4, budget=8, temperature=0.0, saguaro_c=C,
    )
    assert spec == ref, f"C={C}: SSD with Saguaro diverges from greedy"
    assert stats.rounds > 0
