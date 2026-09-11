import numpy as np
import pytest

from policyshiftlab.ranking import (
    compare_model_rankings,
    fixed_candidate_predictions,
    pairwise_reversals,
)


def test_pairwise_reversal_is_detected() -> None:
    target = {"a": 0.10, "b": 0.20, "c": 0.30}
    logged = {"a": 0.25, "b": 0.15, "c": 0.35}

    reversals = pairwise_reversals(target, logged)

    assert len(reversals) == 1
    reversal = reversals[0]
    assert {reversal.model_a, reversal.model_b} == {"a", "b"}
    assert reversal.target_prefers == "a"
    assert reversal.comparison_prefers == "b"


def test_identical_rankings_have_perfect_agreement() -> None:
    y = np.array([0, 0, 1, 1], dtype=float)
    selected = np.ones(4, dtype=bool)
    propensity = np.ones(4, dtype=float)
    predictions = {
        "best": np.array([0.1, 0.2, 0.8, 0.9]),
        "middle": np.array([0.2, 0.3, 0.7, 0.8]),
        "worst": np.full(4, 0.5),
    }

    result = compare_model_rankings(
        y,
        predictions,
        selected,
        propensity,
    )

    assert result.logged_spearman == pytest.approx(1.0)
    assert result.logged_kendall == pytest.approx(1.0)
    assert result.oracle_ipw_spearman == pytest.approx(1.0)
    assert result.oracle_ipw_kendall == pytest.approx(1.0)
    assert result.logged_reversals == ()


def test_logged_ranking_can_reverse_target_order() -> None:
    y = np.array([0, 0, 1, 1, 0, 1], dtype=float)
    selected = np.array([True, True, True, False, False, False])
    propensity = np.full(6, 0.5)
    predictions = {
        "a": np.array([0.45, 0.45, 0.55, 0.95, 0.05, 0.95]),
        "b": np.array([0.05, 0.05, 0.95, 0.45, 0.55, 0.45]),
    }

    result = compare_model_rankings(
        y,
        predictions,
        selected,
        propensity,
    )

    assert result.target_order[0] == "a"
    assert result.logged_order[0] == "b"
    assert len(result.logged_reversals) == 1
    assert result.logged_reversals[0].target_prefers == "a"
    assert result.logged_reversals[0].comparison_prefers == "b"


def test_fixed_candidate_set_is_valid_and_deterministic() -> None:
    p = np.linspace(0.05, 0.95, 20)
    score = np.linspace(-2.0, 2.0, 20)

    first = fixed_candidate_predictions(p, score)
    second = fixed_candidate_predictions(p, score)

    assert set(first) == {
        "oracle",
        "underconfident",
        "overconfident",
        "high_region_robust",
        "low_region_robust",
    }
    for name in first:
        np.testing.assert_allclose(first[name], second[name])
        assert np.all((first[name] > 0.0) & (first[name] < 1.0))


def test_invalid_candidate_shape_is_rejected() -> None:
    with pytest.raises(ValueError):
        compare_model_rankings(
            np.array([0, 1], dtype=float),
            {
                "a": np.array([0.2, 0.8]),
                "b": np.array([0.3]),
            },
            np.array([True, True]),
            np.array([0.5, 0.5]),
        )
