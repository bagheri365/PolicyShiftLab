"""Run Monte Carlo evaluation of selective-observation estimators."""

from policyshiftlab.monte_carlo import run_selection_monte_carlo
from policyshiftlab.synthetic import generate_synthetic_recommender


def main() -> None:
    data = generate_synthetic_recommender(
        n_users=150,
        n_items=80,
        logging_policy="popularity",
        popularity_strength=3.0,
        exposure_strength=1.5,
        min_propensity=0.10,
        max_propensity=0.90,
        seed=0,
    )

    summaries = run_selection_monte_carlo(
        data.outcome,
        data.true_outcome_prob,
        data.propensity,
        n_replications=500,
        seed=123,
    )

    print(
        "estimator      mean       bias       variance    rmse       coverage95"
    )
    for name in ("naive", "stratified", "oracle_ipw"):
        s = summaries[name]
        coverage = (
            f"{s.coverage_95:.3f}"
            if s.coverage_95 == s.coverage_95
            else "n/a"
        )
        print(
            f"{name:<14}"
            f"{s.mean_estimate:>10.6f}"
            f"{s.bias:>11.6f}"
            f"{s.variance:>12.8f}"
            f"{s.rmse:>11.6f}"
            f"{coverage:>12}"
        )

    print(f"\ntarget Brier: {summaries['oracle_ipw'].target:.6f}")


if __name__ == "__main__":
    main()
