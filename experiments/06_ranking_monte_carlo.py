"""Monte Carlo model-ranking stability under repeated selection."""

from policyshiftlab.ranking import fixed_candidate_predictions
from policyshiftlab.ranking_monte_carlo import run_ranking_monte_carlo
from policyshiftlab.synthetic import generate_synthetic_recommender


def _print_summary(summary) -> None:
    print(summary.method)
    print(f"  mean Spearman:            {summary.mean_spearman:.3f}")
    print(f"  mean Kendall:             {summary.mean_kendall:.3f}")
    print(f"  median Spearman:          {summary.median_spearman:.3f}")
    print(f"  median Kendall:           {summary.median_kendall:.3f}")
    print(f"  any-reversal rate:        {summary.any_reversal_rate:.3f}")
    print(f"  mean reversal count:      {summary.mean_reversal_count:.3f}")
    print(
        "  exact-order recovery:     "
        f"{summary.exact_order_recovery_rate:.3f}"
    )


def main() -> None:
    data = generate_synthetic_recommender(
        n_users=300,
        n_items=100,
        logging_policy="popularity",
        popularity_strength=3.0,
        exposure_strength=2.0,
        min_propensity=0.05,
        max_propensity=0.95,
        seed=0,
    )

    predictions = fixed_candidate_predictions(
        data.true_outcome_prob,
        data.logging_score,
    )

    result = run_ranking_monte_carlo(
        data.outcome,
        predictions,
        data.propensity,
        n_replications=500,
        seed=123,
    )

    print(f"replications: {result.logged.n_replications}")
    print()
    _print_summary(result.logged)
    print()
    _print_summary(result.oracle_ipw)


if __name__ == "__main__":
    main()
