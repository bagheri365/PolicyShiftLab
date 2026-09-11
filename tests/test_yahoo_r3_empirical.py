import numpy as np
import pytest

from policyshiftlab.yahoo_r3_empirical import (
    fit_yahoo_probability_models,
    predict_yahoo_probability_models,
    restrict_logged_to_randomized_users,
    run_yahoo_r3_heldout_benchmark,
    split_logged_triplets_by_user,
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


def test_restricts_logged_data_to_randomized_users() -> None:
    train, randomized = _inputs()

    restricted = restrict_logged_to_randomized_users(train, randomized)

    assert set(np.unique(restricted[:, 0])) == {0, 1, 2}
    assert restricted.shape[0] == 12


def test_split_is_reproducible_and_disjoint() -> None:
    train, randomized = _inputs()
    restricted = restrict_logged_to_randomized_users(train, randomized)

    a = split_logged_triplets_by_user(
        restricted, holdout_fraction=0.5, seed=7
    )
    b = split_logged_triplets_by_user(
        restricted, holdout_fraction=0.5, seed=7
    )

    np.testing.assert_array_equal(a.fit_triplets, b.fit_triplets)
    np.testing.assert_array_equal(
        a.evaluation_triplets, b.evaluation_triplets
    )
    fit_pairs = set(map(tuple, a.fit_triplets[:, :2]))
    eval_pairs = set(map(tuple, a.evaluation_triplets[:, :2]))
    assert fit_pairs.isdisjoint(eval_pairs)


def test_model_fitting_and_prediction_are_bounded() -> None:
    train, randomized = _inputs()
    restricted = restrict_logged_to_randomized_users(train, randomized)
    split = split_logged_triplets_by_user(
        restricted, holdout_fraction=0.5, seed=7
    )

    fitted = fit_yahoo_probability_models(split.fit_triplets)
    prediction = predict_yahoo_probability_models(fitted, randomized)

    assert set(prediction) == {
        "global",
        "user_smoothed",
        "item_smoothed",
        "user_item_blend",
    }
    for values in prediction.values():
        assert values.shape == (randomized.shape[0],)
        assert np.all((values >= 0.0) & (values <= 1.0))


def test_unseen_item_falls_back_to_global_component() -> None:
    train, randomized = _inputs()
    restricted = restrict_logged_to_randomized_users(train, randomized)
    fitted = fit_yahoo_probability_models(restricted)

    prediction = predict_yahoo_probability_models(fitted, randomized)

    global_rate = float(fitted["global_rate"])
    np.testing.assert_allclose(prediction["item_smoothed"], global_rate)


def test_benchmark_reports_target_aligned_cohort() -> None:
    train, randomized = _inputs()

    result = run_yahoo_r3_heldout_benchmark(
        train,
        randomized,
        holdout_fraction=0.5,
        seed=7,
    )

    assert result.n_target_users == 3
    assert result.n_items_randomized == 2
    assert result.n_logged_cohort_total == 12
    assert result.n_logged_fit == 6
    assert result.n_logged_evaluation == 6
    assert result.n_randomized == 6
    assert len(result.scores) == 4


def test_randomized_ranking_is_identity_reference() -> None:
    train, randomized = _inputs()

    result = run_yahoo_r3_heldout_benchmark(
        train,
        randomized,
        holdout_fraction=0.5,
        seed=7,
    )
    reference = next(
        row for row in result.rankings if row.method == "randomized"
    )

    assert reference.spearman_vs_randomized == pytest.approx(1.0)
    assert reference.kendall_vs_randomized == pytest.approx(1.0)
    assert reference.reversals_vs_randomized == ()


def test_randomized_labels_do_not_affect_logged_fit() -> None:
    train, randomized = _inputs()
    altered = randomized.copy()
    altered[:, 2] = 5

    original = run_yahoo_r3_heldout_benchmark(
        train, randomized, holdout_fraction=0.5, seed=7
    )
    changed = run_yahoo_r3_heldout_benchmark(
        train, altered, holdout_fraction=0.5, seed=7
    )

    original_logged = {
        row.model: row.heldout_logged_brier for row in original.scores
    }
    changed_logged = {
        row.model: row.heldout_logged_brier for row in changed.scores
    }
    assert original_logged == changed_logged


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.1])
def test_invalid_holdout_fraction_is_rejected(fraction) -> None:
    train, randomized = _inputs()
    restricted = restrict_logged_to_randomized_users(train, randomized)

    with pytest.raises(ValueError):
        split_logged_triplets_by_user(
            restricted, holdout_fraction=fraction
        )
