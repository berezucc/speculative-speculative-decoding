"""Acceptance rate vs cache hit rate Pareto under varying Saguaro C.

Reproduces the spirit of Figure 5 (left) in Kumar et al. 2026: lower C biases
the draft farther from the target, raising hit rate but reducing acceptance.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from _common import PROMPTS, aggregate, base_parser, load, reset_seed_per_run, write_csv
from src.ssd import ssd_decode


def main() -> None:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--Cs", type=float, nargs="+", default=[1.0, 0.7, 0.5, 0.3, 0.1, 0.01])
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    draft, verifier = load(args.draft, args.verifier)

    rows = []
    for C in args.Cs:
        hit_rates, acc_rates = [], []
        for prompt in PROMPTS[: args.prompts]:
            reset_seed_per_run()
            _, stats = ssd_decode(
                draft, verifier, prompt, n_tokens=args.n_tokens,
                k=args.k, budget=args.budget, temperature=args.temperature, saguaro_c=C,
            )
            hit_rates.append(stats.hit_rate)
            acc_rates.append(stats.acceptance_rate)
        row = {
            "C": C,
            "hit_rate": aggregate(hit_rates),
            "acceptance_rate": aggregate(acc_rates),
        }
        rows.append(row)
        print(f"C={C:.2f}  hit_rate={row['hit_rate']:.2%}  α={row['acceptance_rate']:.2%}")

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "acceptance_hit_pareto.csv", rows, ["C", "hit_rate", "acceptance_rate"])

    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(
        [r["hit_rate"] for r in rows],
        [r["acceptance_rate"] for r in rows],
        c=[r["C"] for r in rows], cmap="viridis", s=80,
    )
    for r in rows:
        ax.annotate(f"C={r['C']}", (r["hit_rate"], r["acceptance_rate"]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    plt.colorbar(sc, ax=ax, label="C")
    ax.set_xlabel("cache hit rate")
    ax.set_ylabel("acceptance rate α")
    ax.grid(alpha=0.3)
    ax.set_title(f"Saguaro sampling Pareto  (K={args.k}, B={args.budget}, T={args.temperature})")
    fig.tight_layout()
    fig.savefig(args.out / "acceptance_hit_pareto.png", dpi=150)
    print(f"wrote {args.out / 'acceptance_hit_pareto.png'}")


if __name__ == "__main__":
    main()
