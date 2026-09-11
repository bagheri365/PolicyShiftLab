"""Bootstrap uncertainty for the top two Coat candidate models."""

from pathlib import Path

from policyshiftlab.coat_audit import load_coat_matrix
from policyshiftlab.coat_repeated import bootstrap_randomized_pair_difference


def main() -> None:
    base = Path("data/coat")
    train_path = base / "train.ascii"
    randomized_path = base / "test.ascii"

    missing = [
        str(path)
        for path in (train_path, randomized_path)
        if not path.exists()
    ]
    if missing:
        raise SystemExit("Missing Coat files: " + ", ".join(missing))

    result = bootstrap_randomized_pair_difference(
        load_coat_matrix(train_path),
        load_coat_matrix(randomized_path),
        fit_seed=2026,
        holdout_fraction=1.0 / 3.0,
        positive_threshold=4,
        prior_strength=10.0,
        model_a="user_item_blend",
        model_b="user_smoothed",
        n_bootstrap=5000,
        seed=271828,
    )

    print("Coat randomized benchmark: top-pair user-cluster bootstrap")
    print(
        f"difference = Brier({result.model_a}) - "
        f"Brier({result.model_b})"
    )
    print(f"observed difference: {result.observed_difference:+.6f}")
    print(
        "pointwise 95% percentile interval: "
        f"[{result.lower_95:+.6f}, {result.upper_95:+.6f}]"
    )
    print(
        f"bootstrap P({result.model_a} better): "
        f"{result.probability_a_better:.3f}"
    )
    print(f"bootstrap replicates: {result.n_bootstrap}")
    print(f"bootstrap seed: {result.seed}")
    print()
    print(
        "Negative differences favor user_item_blend. Users are resampled as "
        "clusters, carrying all randomized observations for a sampled user "
        "together. This is descriptive uncertainty for the fixed seed-2026 "
        "fitted models, not a formal hypothesis test."
    )


if __name__ == "__main__":
    main()
