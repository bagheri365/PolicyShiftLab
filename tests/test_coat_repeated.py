import numpy as np
import pytest

from policyshiftlab.coat_repeated import (
    bootstrap_randomized_pair_difference,
    run_repeated_coat_holdouts,
)


def _inputs():
    train = np.array(
        [
            [5, 3, 4, 2, 0, 0],
            [1, 5, 2, 4, 0, 0],
            [5, 1, 4, 3, 0, 0],
            [2, 4, 5, 1, 0, 0],
        ],
        dtype=int,
    )
    randomized = np.array(
        [
            [0, 0, 0, 0, 5, 2],
            [0, 0, 0, 0, 3, 4],
            [0, 0, 0, 0, 4, 1],
            [0, 0, 0, 0, 2, 5],
        ],
        dtype=int,
    )
    propensity = np.array(
        [
            [0.5, 0.3, 0.4, 0.2, 0.1, 0.1],
            [0.2, 0.5, 0.3, 0.4, 0.1, 0.1],
            [0.5, 0.2, 0.4, 0.3, 0.1, 0.1],
            [0.3, 0.4, 0.5, 0.2, 0.1, 0.1],
        ],
        dtype=float,
    )
    return train, randomized, propensity


def test_repeated_summary_has_three_primary_methods() -> None:
    train, randomized, propensity = _inputs()
    result = run_repeated_coat_holdouts(
        train, randomized, propensity,
        n_splits=5, first_seed=10, holdout_fraction=0.5,
    )
    assert result.n_splits == 5
    assert {row.method for row in result.method_summaries} == {
        "heldout_logged_naive", "estimated_ht", "estimated_snips"
    }


def test_repeated_summary_rates_are_bounded() -> None:
    train, randomized, propensity = _inputs()
    result = run_repeated_coat_holdouts(
        train, randomized, propensity,
        n_splits=4, holdout_fraction=0.5,
    )
    for row in result.method_summaries:
        assert 0.0 <= row.any_reversal_rate <= 1.0
        assert 0.0 <= row.exact_order_recovery_rate <= 1.0
    for row in result.pair_reversal_summaries:
        assert 0.0 <= row.reversal_rate <= 1.0


def test_repeated_summary_is_deterministic() -> None:
    train, randomized, propensity = _inputs()
    a = run_repeated_coat_holdouts(
        train, randomized, propensity,
        n_splits=4, first_seed=2, holdout_fraction=0.5,
    )
    b = run_repeated_coat_holdouts(
        train, randomized, propensity,
        n_splits=4, first_seed=2, holdout_fraction=0.5,
    )
    assert a.n_splits == b.n_splits
    assert a.first_seed == b.first_seed
    for ra, rb in zip(a.method_summaries, b.method_summaries):
        assert ra.method == rb.method
        np.testing.assert_allclose(
            [
                ra.mean_signed_brier_error,
                ra.mean_absolute_brier_error,
                ra.rmse_brier_error,
                ra.mean_spearman,
                ra.median_spearman,
                ra.mean_kendall,
                ra.median_kendall,
                ra.any_reversal_rate,
                ra.mean_reversal_count,
                ra.exact_order_recovery_rate,
            ],
            [
                rb.mean_signed_brier_error,
                rb.mean_absolute_brier_error,
                rb.rmse_brier_error,
                rb.mean_spearman,
                rb.median_spearman,
                rb.mean_kendall,
                rb.median_kendall,
                rb.any_reversal_rate,
                rb.mean_reversal_count,
                rb.exact_order_recovery_rate,
            ],
            equal_nan=True,
        )


def test_bootstrap_is_deterministic_and_ordered() -> None:
    train, randomized, _ = _inputs()
    a = bootstrap_randomized_pair_difference(
        train, randomized, fit_seed=7, holdout_fraction=0.5,
        n_bootstrap=100, seed=9,
    )
    b = bootstrap_randomized_pair_difference(
        train, randomized, fit_seed=7, holdout_fraction=0.5,
        n_bootstrap=100, seed=9,
    )
    assert a == b
    assert a.lower_95 <= a.upper_95
    assert 0.0 <= a.probability_a_better <= 1.0


@pytest.mark.parametrize("n_splits", [0, -1])
def test_invalid_repeated_count_is_rejected(n_splits) -> None:
    train, randomized, propensity = _inputs()
    with pytest.raises(ValueError):
        run_repeated_coat_holdouts(
            train, randomized, propensity, n_splits=n_splits
        )


def test_invalid_bootstrap_count_is_rejected() -> None:
    train, randomized, _ = _inputs()
    with pytest.raises(ValueError):
        bootstrap_randomized_pair_difference(
            train, randomized, n_bootstrap=0
        )
