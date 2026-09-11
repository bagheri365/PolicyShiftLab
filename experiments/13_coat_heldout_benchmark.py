"""Run leakage-safe held-out logged vs randomized Coat evaluation."""

from pathlib import Path

from policyshiftlab.coat_audit import load_coat_matrix
from policyshiftlab.coat_heldout import run_coat_heldout_benchmark
from policyshiftlab.coat_propensity_audit import load_coat_propensities


def main() -> None:
    base = Path("data/coat")
    train_path = base / "train.ascii"
    randomized_path = base / "test.ascii"
    propensity_path = base / "propensities.ascii"

    missing = [
        str(path)
        for path in (train_path, randomized_path, propensity_path)
        if not path.exists()
    ]
    if missing:
        raise SystemExit("Missing Coat files: " + ", ".join(missing))

    result = run_coat_heldout_benchmark(
        load_coat_matrix(train_path),
        load_coat_matrix(randomized_path),
        load_coat_propensities(propensity_path),
        positive_threshold=4,
        prior_strength=10.0,
        holdout_fraction=1.0 / 3.0,
        seed=2026,
        propensity_floors=(0.01, 0.02, 0.05),
    )

    print(
        f"Coat held-out benchmark: "
        f"{result.n_users} users x {result.n_items} items"
    )
    print(f"logged total:       {result.n_logged_total}")
    print(f"logged fit:         {result.n_logged_fit}")
    print(f"logged evaluation:  {result.n_logged_evaluation}")
    print(f"randomized:         {result.n_randomized}")
    print(
        f"holdout fraction:   {result.holdout_fraction:.3f} "
        f"(seed={result.seed})"
    )
    print(f"binary event:       rating >= {result.positive_threshold}")
    print()
    print(
        "model                 randomized   heldout-log      est-HT      SNIPS"
    )
    for row in result.scores:
        print(
            f"{row.model:20s} "
            f"{row.randomized_brier:10.6f} "
            f"{row.heldout_logged_brier:12.6f} "
            f"{row.estimated_ht_brier:10.6f} "
            f"{row.estimated_snips_brier:10.6f}"
        )

    print()
    print("ranking agreement with randomized benchmark")
    for row in result.rankings:
        print(
            f"{row.method:22s} "
            f"Spearman={row.spearman_vs_randomized:6.3f} "
            f"Kendall={row.kendall_vs_randomized:6.3f} "
            f"reversals={len(row.reversals_vs_randomized)} "
            f"order={' > '.join(row.order)}"
        )

    print()
    print("propensity-floor sensitivity")
    floors = sorted({row.propensity_floor for row in result.clipping})
    for floor in floors:
        print(f"floor={floor:.2f}")
        for row in result.clipping:
            if row.propensity_floor != floor:
                continue
            print(
                f"  {row.model:18s} "
                f"clipped-HT={row.estimated_ht_brier:.6f} "
                f"clipped-SNIPS={row.estimated_snips_brier:.6f}"
            )

    print()
    print(
        "Models are fitted only on the logged-fit subset. Held-out logged "
        "evaluation uses separate logged ratings, and randomized ratings remain "
        "evaluation-only. Estimated HT uses supplied exposure propensity times "
        "the known userwise holdout probability. Clipped estimates are "
        "sensitivity diagnostics, not unbiased HT estimates for the original "
        "target estimand."
    )


if __name__ == "__main__":
    main()
