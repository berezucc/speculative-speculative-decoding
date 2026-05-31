"""Shared helpers for benchmark scripts."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from statistics import mean
from typing import Iterable

# Make `src` importable when running scripts directly from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import LM, assert_shared_vocab  # noqa: E402
from src.utils import get_device, set_seed  # noqa: E402


PROMPTS = [
    "The transformer architecture",
    "def fibonacci(n):\n    if n < 2:",
    "Q: What is the capital of France?\nA:",
    "Once upon a time in a faraway land,",
]


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--draft", default="distilgpt2")
    p.add_argument("--verifier", default="gpt2-medium")
    p.add_argument("--n_tokens", type=int, default=30)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--prompts", type=int, default=len(PROMPTS), help="number of prompts to average over")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "docs" / "figures")
    return p


def load(draft_name: str, verifier_name: str):
    device = get_device()
    print(f"loading draft={draft_name} verifier={verifier_name} on {device}", flush=True)
    draft = LM.load(draft_name, device)
    verifier = LM.load(verifier_name, device)
    assert_shared_vocab(draft, verifier)
    return draft, verifier


def aggregate(values: Iterable[float]) -> float:
    vs = list(values)
    return mean(vs) if vs else 0.0


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def reset_seed_per_run():
    set_seed(42)
