"""Run the overlap stress test."""

from policyshiftlab.overlap import run_overlap_stress_test


def main() -> None:
    results = run_overlap_stress_test(
        min_propensities=(0.20, 0.10, 0.05, 0.02),
        exposure_strengths=(1.0, 1.5, 2.0, 3.0),
        n_users=120,
        n_items=60,
        n_replications=300,
        seed=0,
    )

    print(
        "min_p  strength  mean_p   ESS_exp   "
        "naive_rmse  strat_rmse  ipw_rmse  ipw_bias"
    )

    for r in results:
        print(
            f"{r.min_propensity:>5.2f}"
            f"{r.exposure_strength:>10.2f}"
            f"{r.mean_propensity:>9.3f}"
            f"{r.mean_ipw_effective_sample_size:>10.1f}"
            f"{r.naive_rmse:>12.6f}"
            f"{r.stratified_rmse:>12.6f}"
            f"{r.oracle_ipw_rmse:>10.6f}"
            f"{r.oracle_ipw_bias:>10.6f}"
        )


if __name__ == "__main__":
    main()
