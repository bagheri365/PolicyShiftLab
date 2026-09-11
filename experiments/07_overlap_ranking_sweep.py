"""Sweep overlap strength and summarize ranking stability."""

from policyshiftlab.ranking_overlap import run_ranking_overlap_sweep


def main() -> None:
    results = run_ranking_overlap_sweep(
        regimes=(
            ("strong", 0.20, 1.0),
            ("moderate", 0.10, 1.5),
            ("weak", 0.02, 3.0),
        ),
        n_users=200,
        n_items=80,
        n_replications=300,
        seed=0,
    )

    print(
        "regime     ESS_exp   "
        "log_sp  log_tau  log_rev  log_exact   "
        "ipw_sp  ipw_tau  ipw_rev  ipw_exact"
    )

    for r in results:
        print(
            f"{r.label:<10}"
            f"{r.expected_ipw_effective_sample_size:>8.1f}   "
            f"{r.logged_mean_spearman:>6.3f}  "
            f"{r.logged_mean_kendall:>7.3f}  "
            f"{r.logged_any_reversal_rate:>7.3f}  "
            f"{r.logged_exact_order_recovery_rate:>9.3f}   "
            f"{r.ipw_mean_spearman:>6.3f}  "
            f"{r.ipw_mean_kendall:>7.3f}  "
            f"{r.ipw_any_reversal_rate:>7.3f}  "
            f"{r.ipw_exact_order_recovery_rate:>9.3f}"
        )


if __name__ == "__main__":
    main()
