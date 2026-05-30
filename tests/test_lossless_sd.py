"""SD at temperature 0 must produce token-identical output to verifier-only greedy."""
import pytest

from src.models import LM, assert_shared_vocab
from src.sd import greedy_decode, speculative_decode
from src.utils import get_device, set_seed


@pytest.fixture(scope="module")
def models():
    device = get_device()
    draft = LM.load("distilgpt2", device)
    verifier = LM.load("distilgpt2", device)  # self-speculation: trivially lossless, fast
    assert_shared_vocab(draft, verifier)
    return draft, verifier


@pytest.mark.parametrize("k", [1, 4, 8])
def test_sd_matches_greedy_at_temp_zero(models, k):
    draft, verifier = models
    prompt = "The transformer architecture"
    n = 20

    set_seed()
    ref = greedy_decode(verifier, prompt, n)
    set_seed()
    spec, _ = speculative_decode(draft, verifier, prompt, n, k=k, temperature=0.0)
    assert spec == ref, f"K={k}: divergence between SD and greedy"
