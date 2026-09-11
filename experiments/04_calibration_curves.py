"""Plot target, logged, and oracle-IPW reliability curves."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from policyshiftlab.calibration import (
    bootstrap_calibration_bands,
    calibration_bins,
    expected_calibration_error,
    quantile_bin_edges,
)
from policyshiftlab.synthetic import generate_synthetic_recommender


def main() -> None:
    data = generate_synthetic_recommender(
        n_users=500,
        n_items=150,
        logging_policy="popularity",
        popularity_strength=3.0,
        exposure_strength=2.0,
        min_propensity=0.05,
        max_propensity=0.95,
        seed=0,
    )

    y = data.outcome
    p = data.true_outcome_prob
    selected = data.selected
    ipw = 1.0 / data.propensity[selected]

    # Derive score intervals once from the target population and reuse them.
    # This keeps the three reliability curves directly comparable.
    edges = quantile_bin_edges(p, n_bins=10)

    target_bins = calibration_bins(
        y,
        p,
        bin_edges=edges,
    )
    logged_bins = calibration_bins(
        y[selected],
        p[selected],
        bin_edges=edges,
    )
    ipw_bins = calibration_bins(
        y[selected],
        p[selected],
        sample_weight=ipw,
        bin_edges=edges,
    )

    target_ece = expected_calibration_error(
        y,
        p,
        bin_edges=edges,
    )
    logged_ece = expected_calibration_error(
        y[selected],
        p[selected],
        bin_edges=edges,
    )
    ipw_ece = expected_calibration_error(
        y[selected],
        p[selected],
        sample_weight=ipw,
        bin_edges=edges,
    )

    target_band = bootstrap_calibration_bands(
        y,
        p,
        bin_edges=edges,
        n_bootstrap=300,
        seed=101,
    )
    logged_band = bootstrap_calibration_bands(
        y[selected],
        p[selected],
        bin_edges=edges,
        n_bootstrap=300,
        seed=102,
    )
    ipw_band = bootstrap_calibration_bands(
        y[selected],
        p[selected],
        sample_weight=ipw,
        bin_edges=edges,
        n_bootstrap=300,
        seed=103,
    )

    figure_dir = Path("figures")
    figure_dir.mkdir(exist_ok=True)
    output_path = figure_dir / "target_logged_ipw_reliability.png"

    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.plot(
        target_bins.mean_predicted,
        target_bins.fraction_positive,
        marker="o",
        label=f"Target (ECE={target_ece:.3f})",
    )
    plt.fill_between(
        target_band.mean_predicted,
        target_band.lower,
        target_band.upper,
        alpha=0.12,
    )
    plt.plot(
        logged_bins.mean_predicted,
        logged_bins.fraction_positive,
        marker="o",
        label=f"Logged (ECE={logged_ece:.3f})",
    )
    plt.fill_between(
        logged_band.mean_predicted,
        logged_band.lower,
        logged_band.upper,
        alpha=0.12,
    )
    plt.plot(
        ipw_bins.mean_predicted,
        ipw_bins.fraction_positive,
        marker="o",
        label=f"Oracle IPW (ECE={ipw_ece:.3f})",
    )
    plt.fill_between(
        ipw_band.mean_predicted,
        ipw_band.lower,
        ipw_band.upper,
        alpha=0.12,
    )
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive rate")
    plt.title("Reliability under policy-induced selective observation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print(f"saved: {output_path}")
    print(f"shared bins: {len(edges) - 1}")
    print(f"target ECE: {target_ece:.6f}")
    print(f"logged ECE: {logged_ece:.6f}")
    print(f"oracle IPW ECE: {ipw_ece:.6f}")
    print("bands: pointwise 95% percentile bootstrap intervals")
    print(
        "logged raw count range: "
        f"{logged_bins.bin_count.min()}-{logged_bins.bin_count.max()}"
    )
    print(
        "IPW weighted-mass range: "
        f"{np.min(ipw_bins.bin_weight):.1f}-{np.max(ipw_bins.bin_weight):.1f}"
    )


if __name__ == "__main__":
    main()
