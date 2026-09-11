import numpy as np
import pytest

from policyshiftlab.coat_heldout import (
    run_coat_heldout_benchmark,
    split_logged_ratings_by_user,
)


def _inputs():
    train = np.array(
        [
            [5, 3, 4, 2, 0, 0],
            [1, 5, 2, 4, 0, 0],
            [5, 1, 4, 3, 0, 0],
        ],
        dtype=int,
    )
    randomized = np.array(
        [
            [0, 0, 0, 0, 5, 2],
            [0, 0, 0, 0, 3, 4],
            [0, 0, 0, 0, 4, 1],
        ],
        dtype=int,
    )
    propensity = np.array(
        [
            [0.5, 0.3, 0.4, 0.2, 0.1, 0.1],
            [0.2, 0.5, 0.3, 0.4, 0.1, 0.1],
            [0.5, 0.2, 0.4, 0.3, 0.1, 0.1],
        ],
        dtype=float,
    )
    return train, randomized, propensity


def test_split_is_reproducible_and_disjoint() -> None:
    train, _, _ = _inputs()

    a = split_logged_ratings_by_user(
        train,
        holdout_fraction=0.5,
        seed=7,
    )
    b = split_logged_ratings_by_user(
        train,
        holdout_fraction=0.5,
        seed=7,
    )

    np.testing.assert_array_equal(a.fit_matrix, b.fit_matrix)
    np.testing.assert_array_equal(a.evaluation_mask, b.evaluation_mask)
    assert not np.any((a.fit_matrix > 0) & a.evaluation_mask)
    assert a.n_fit + a.n_evaluation == int(np.sum(train > 0))


def test_split_has_known_userwise_holdout_probability() -> None:
    train, _, _ = _inputs()

    split = split_logged_ratings_by_user(
        train,
        holdout_fraction=0.5,
        seed=7,
    )

    observed = train > 0
    for user in range(train.shape[0]):
        probs = split.evaluation_probability[user, observed[user]]
        np.testing.assert_allclose(probs, 0.5)


def test_benchmark_fits_only_on_logged_fit_subset() -> None:
    train, randomized, propensity = _inputs()

    result = run_coat_heldout_benchmark(
        train,
        randomized,
        propensity,
        holdout_fraction=0.5,
        seed=7,
        propensity_floors=(0.1,),
    )

    assert result.n_logged_total == 12
    assert result.n_logged_fit == 6
    assert result.n_logged_evaluation == 6
    assert result.n_randomized == 6
    assert len(result.scores) == 4


def test_randomized_labels_do_not_affect_logged_split() -> None:
    train, randomized, propensity = _inputs()
    altered = randomized.copy()
    altered[altered > 0] = 5

    original = run_coat_heldout_benchmark(
        train,
        randomized,
        propensity,
        holdout_fraction=0.5,
        seed=11,
        propensity_floors=(0.1,),
    )
    changed = run_coat_heldout_benchmark(
        train,
        altered,
        propensity,
        holdout_fraction=0.5,
        seed=11,
        propensity_floors=(0.1,),
    )

    original_nonrandomized = {
        row.model: (
            row.heldout_logged_brier,
            row.estimated_ht_brier,
            row.estimated_snips_brier,
        )
        for row in original.scores
    }
    changed_nonrandomized = {
        row.model: (
            row.heldout_logged_brier,
            row.estimated_ht_brier,
            row.estimated_snips_brier,
        )
        for row in changed.scores
    }
    assert original_nonrandomized == changed_nonrandomized


def test_randomized_reference_ranking_is_identity() -> None:
    train, randomized, propensity = _inputs()

    result = run_coat_heldout_benchmark(
        train,
        randomized,
        propensity,
        holdout_fraction=0.5,
        seed=7,
        propensity_floors=(0.1,),
    )
    reference = next(
        row for row in result.rankings if row.method == "randomized"
    )

    assert reference.spearman_vs_randomized == pytest.approx(1.0)
    assert reference.kendall_vs_randomized == pytest.approx(1.0)
    assert reference.reversals_vs_randomized == ()


def test_clipping_does_not_modify_primary_scores() -> None:
    train, randomized, propensity = _inputs()

    a = run_coat_heldout_benchmark(
        train,
        randomized,
        propensity,
        holdout_fraction=0.5,
        seed=7,
        propensity_floors=(0.05,),
    )
    b = run_coat_heldout_benchmark(
        train,
        randomized,
        propensity,
        holdout_fraction=0.5,
        seed=7,
        propensity_floors=(0.2,),
    )

    assert a.scores == b.scores


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.1])
def test_invalid_holdout_fraction_is_rejected(fraction) -> None:
    train, _, _ = _inputs()

    with pytest.raises(ValueError):
        split_logged_ratings_by_user(
            train,
            holdout_fraction=fraction,
        )
