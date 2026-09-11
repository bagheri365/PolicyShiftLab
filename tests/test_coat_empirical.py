import numpy as np
import pytest

from policyshiftlab.coat_empirical import (
    fit_logged_probability_models,
    run_coat_empirical_benchmark,
)


def _inputs():
    train = np.array(
        [
            [5, 0, 4, 0],
            [0, 2, 0, 5],
            [1, 0, 5, 0],
        ],
        dtype=int,
    )
    randomized = np.array(
        [
            [0, 4, 0, 2],
            [5, 0, 3, 0],
            [0, 5, 0, 1],
        ],
        dtype=int,
    )
    propensity = np.array(
        [
            [0.5, 0.1, 0.4, 0.2],
            [0.2, 0.3, 0.1, 0.5],
            [0.25, 0.15, 0.4, 0.2],
        ],
        dtype=float,
    )
    return train, randomized, propensity


def test_model_fitting_uses_only_train_argument() -> None:
    train, _, _ = _inputs()

    predictions = fit_logged_probability_models(train)

    assert set(predictions) == {
        "global",
        "user_smoothed",
        "item_smoothed",
        "user_item_blend",
    }
    for prediction in predictions.values():
        assert prediction.shape == train.shape
        assert np.all((prediction >= 0.0) & (prediction <= 1.0))


def test_global_model_equals_logged_positive_rate() -> None:
    train, _, _ = _inputs()

    predictions = fit_logged_probability_models(train, positive_threshold=4)

    observed = train > 0
    expected = np.mean(train[observed] >= 4)
    np.testing.assert_allclose(predictions["global"], expected)


def test_benchmark_reports_all_methods_and_models() -> None:
    train, randomized, propensity = _inputs()

    result = run_coat_empirical_benchmark(
        train,
        randomized,
        propensity,
        propensity_floors=(0.1, 0.2),
    )

    assert result.n_logged == 6
    assert result.n_randomized == 6
    assert len(result.scores) == 4
    assert {row.method for row in result.rankings} == {
        "randomized",
        "logged_naive",
        "estimated_ht",
        "estimated_snips",
    }
    assert len(result.clipping) == 8


def test_randomized_ranking_is_identity_reference() -> None:
    train, randomized, propensity = _inputs()

    result = run_coat_empirical_benchmark(train, randomized, propensity)
    randomized_summary = next(
        row for row in result.rankings if row.method == "randomized"
    )

    assert randomized_summary.spearman_vs_randomized == pytest.approx(1.0)
    assert randomized_summary.kendall_vs_randomized == pytest.approx(1.0)
    assert randomized_summary.reversals_vs_randomized == ()


def test_randomized_labels_do_not_change_fitted_predictions() -> None:
    train, randomized, propensity = _inputs()
    altered = randomized.copy()
    altered[altered > 0] = 5

    original = fit_logged_probability_models(train)
    _ = run_coat_empirical_benchmark(train, altered, propensity)
    refit = fit_logged_probability_models(train)

    for name in original:
        np.testing.assert_allclose(original[name], refit[name])


def test_floor_sensitivity_does_not_modify_untrimmed_scores() -> None:
    train, randomized, propensity = _inputs()

    a = run_coat_empirical_benchmark(
        train,
        randomized,
        propensity,
        propensity_floors=(0.05,),
    )
    b = run_coat_empirical_benchmark(
        train,
        randomized,
        propensity,
        propensity_floors=(0.2,),
    )

    assert a.scores == b.scores


@pytest.mark.parametrize(
    "propensity",
    [
        np.ones((3, 3)),
        np.zeros((3, 4)),
        np.ones((3, 4)) * 1.1,
    ],
)
def test_invalid_propensity_inputs_are_rejected(propensity) -> None:
    train, randomized, _ = _inputs()

    with pytest.raises(ValueError):
        run_coat_empirical_benchmark(train, randomized, propensity)
