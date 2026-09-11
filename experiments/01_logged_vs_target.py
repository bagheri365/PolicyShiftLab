"""Run the first logged-vs-target Brier evaluation experiment."""

from policyshiftlab.evaluation import evaluate_brier_under_selection
from policyshiftlab.synthetic import generate_synthetic_recommender


def main() -> None:
    data = generate_synthetic_recommender(
        n_users=500,
        n_items=150,
        logging_policy="popularity",
        popularity_strength=3.0,
        exposure_strength=2.0,
        seed=0,
    )

    result = evaluate_brier_under_selection(
        data.outcome,
        data.true_outcome_prob,
        data.selected,
        data.propensity,
    )

    print(f"pairs:                 {data.n_pairs}")
    print(f"selected:              {data.n_selected}")
    print(f"target Brier:          {result.target:.6f}")
    print(f"naive logged Brier:    {result.logged_naive:.6f}")
    print(f"oracle IPW Brier:      {result.logged_oracle_ipw:.6f}")
    print(f"stratified Brier:      {result.logged_stratified:.6f}")
    print(f"IPW effective N:       {result.ipw_effective_sample_size:.1f}")


if __name__ == "__main__":
    main()
