"""Uniform vs geometric fan-out: hit rate at several budgets.

Reproduces the spirit of Figure 4 in Kumar et al. 2026.
"""
from __future__ import annotations

from functools import partial

import matplotlib.pyplot as plt

from _common import PROMPTS, aggregate, base_parser, load, reset_seed_per_run, write_csv
from src.outcomes import geometric_fan_out, uniform_fan_out
from src.ssd import ssd_decode


def main() -> None:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--a_p", type=float, default=0.7, help="assumed draft acceptance rate for geometric")
    parser.add_argument("--r", type=float, default=1.0)
    args = parser.parse_args()

    draft, verifier = load(args.draft, args.verifier)

    rows = []
    for B in args.budgets:
        for label, fn in [
            ("uniform", uniform_fan_out),
            ("geometric", partial(geometric_fan_out, acceptance_rate=args.a_p, r=args.r)),
        ]:
            hit_rates = []
            for prompt in PROMPTS[: args.prompts]:
                reset_seed_per_run()
                _, stats = ssd_decode(
                    draft, verifier, prompt, n_tokens=args.n_tokens,
                    k=args.k, budget=B, temperature=args.temperature, fan_out_fn=fn,
                )
                hit_rates.append(stats.hit_rate)
            rows.append({"budget": B, "strategy": label, "hit_rate": aggregate(hit_rates)})
            print(f"B={B:3d}  {label:10s}  hit_rate={rows[-1]['hit_rate']:.2%}")

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "fan_out_sweep.csv", rows, ["budget", "strategy", "hit_rate"])

    fig, ax = plt.subplots(figsize=(6, 4))
    for label in ["uniform", "geometric"]:
        xs = [r["budget"] for r in rows if r["strategy"] == label]
        ys = [r["hit_rate"] for r in rows if r["strategy"] == label]
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("fan-out budget B")
    ax.set_ylabel("cache hit rate")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(f"fan-out strategy  (K={args.k}, T={args.temperature})")
    fig.tight_layout()
    fig.savefig(args.out / "fan_out_sweep.png", dpi=150)
    print(f"wrote {args.out / 'fan_out_sweep.png'}")


if __name__ == "__main__":
    main()
