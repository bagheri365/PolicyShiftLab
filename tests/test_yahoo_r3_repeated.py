import numpy as np
import pytest

from policyshiftlab.yahoo_r3_repeated import (
    bootstrap_yahoo_randomized_pair_difference,
    run_repeated_yahoo_r3_holdouts,
)


def _inputs():
    train = np.array(
        [
            [0, 0, 5], [0, 1, 3], [0, 2, 4], [0, 3, 2],
            [1, 0, 1], [1, 1, 5], [1, 2, 2], [1, 3, 4],
            [2, 0, 5], [2, 1, 1], [2, 2, 4], [2, 3, 3],
            [3, 0, 5], [3, 1, 5], [3, 2, 5], [3, 3, 5],
        ],
        dtype=int,
    )
    randomized = np.array(
        [
            [0, 4, 5], [0, 5, 2],
            [1, 4, 3], [1, 5, 4],
            [2, 4, 4], [2, 5, 1],
        ],
        dtype=int,
    )
    return train, randomized


def test_repeated_summary_is_deterministic() -> None:
    train, randomized = _inputs()

    a = run_repeated_yahoo_r3_holdouts(
        train,
        randomized,
        n_splits=5,
        first_seed=3,
        holdout_fraction=0.5,
    )
    b = run_repeated_yahoo_r3_holdouts(
        train,
        randomized,
        n_splits=5,
        first_seed=3,
        holdout_fraction=0.5,
    )

    assert a == b


def test_repeated_summary_rates_are_bounded() -> None:
    train, randomized = _inputs()

    result = run_repeated_yahoo_r3_holdouts(
        train,
        randomized,
        n_splits=4,
        holdout_fraction=0.5,
    )

    row = result.method_summary
    assert 0.0 <= row.any_reversal_rate <= 1.0
    assert 0.0 <= row.exact_order_recovery_rate <= 1.0
    assert 0.0 <= result.pair_reversal.reversal_rate <= 1.0


def test_repeated_summary_counts_splits() -> None:
    train, randomized = _inputs()

    result = run_repeated_yahoo_r3_holdouts(
        train,
        randomized,
        n_splits=6,
        first_seed=10,
        holdout_fraction=0.5,
    )

    assert result.n_splits == 6
    assert result.first_seed == 10
    assert result.method_summary.n_splits == 6
    assert result.pair_reversal.n_splits == 6


def test_bootstrap_is_deterministic_and_ordered() -> None:
    train, randomized = _inputs()

    a = bootstrap_yahoo_randomized_pair_difference(
        train,
        randomized,
        fit_seed=7,
        holdout_fraction=0.5,
        n_bootstrap=100,
        seed=9,
    )
    b = bootstrap_yahoo_randomized_pair_difference(
        train,
        randomized,
        fit_seed=7,
        holdout_fraction=0.5,
        n_bootstrap=100,
        seed=9,
    )

    assert a == b
    assert a.lower_95 <= a.upper_95
    assert 0.0 <= a.probability_a_better <= 1.0


@pytest.mark.parametrize("n_splits", [0, -1])
def test_invalid_split_count_is_rejected(n_splits) -> None:
    train, randomized = _inputs()

    with pytest.raises(ValueError):
        run_repeated_yahoo_r3_holdouts(
            train,
            randomized,
            n_splits=n_splits,
        )


def test_invalid_bootstrap_count_is_rejected() -> None:
    train, randomized = _inputs()

    with pytest.raises(ValueError):
        bootstrap_yahoo_randomized_pair_difference(
            train,
            randomized,
            n_bootstrap=0,
        )
