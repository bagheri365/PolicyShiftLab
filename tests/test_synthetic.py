import numpy as np
import pytest

from policyshiftlab.synthetic import generate_synthetic_recommender


def test_synthetic_recommender_shapes() -> None:
    data = generate_synthetic_recommender(
        n_users=12,
        n_items=7,
        latent_dim=3,
        seed=7,
    )

    expected_pairs = 12 * 7

    assert data.n_pairs == expected_pairs
    assert data.user_id.shape == (expected_pairs,)
    assert data.item_id.shape == (expected_pairs,)
    assert data.true_outcome_prob.shape == (expected_pairs,)
    assert data.outcome.shape == (expected_pairs,)
    assert data.propensity.shape == (expected_pairs,)
    assert data.selected.shape == (expected_pairs,)


def test_probabilities_and_propensities_are_valid() -> None:
    data = generate_synthetic_recommender(seed=11)

    assert np.all((data.true_outcome_prob > 0.0) & (data.true_outcome_prob < 1.0))
    assert np.all((data.propensity >= 0.05) & (data.propensity <= 0.95))
    assert set(np.unique(data.outcome)).issubset({0, 1})
    assert data.selected.dtype == np.bool_


def test_generation_is_reproducible_for_fixed_seed() -> None:
    first = generate_synthetic_recommender(seed=123)
    second = generate_synthetic_recommender(seed=123)

    np.testing.assert_array_equal(first.user_id, second.user_id)
    np.testing.assert_array_equal(first.item_id, second.item_id)
    np.testing.assert_allclose(first.true_outcome_prob, second.true_outcome_prob)
    np.testing.assert_array_equal(first.outcome, second.outcome)
    np.testing.assert_allclose(first.propensity, second.propensity)
    np.testing.assert_array_equal(first.selected, second.selected)


def test_different_seed_changes_population() -> None:
    first = generate_synthetic_recommender(seed=1)
    second = generate_synthetic_recommender(seed=2)

    assert not np.allclose(first.true_outcome_prob, second.true_outcome_prob)


def test_popularity_policy_changes_logging_score_and_propensity() -> None:
    aligned = generate_synthetic_recommender(
        logging_policy="aligned",
        seed=42,
    )
    popularity = generate_synthetic_recommender(
        logging_policy="popularity",
        popularity_strength=2.0,
        seed=42,
    )

    assert not np.allclose(aligned.logging_score, popularity.logging_score)
    assert not np.allclose(aligned.propensity, popularity.propensity)


def test_popularity_policy_increases_popularity_exposure_association() -> None:
    aligned = generate_synthetic_recommender(
        logging_policy="aligned",
        n_users=300,
        n_items=80,
        seed=9,
    )
    distorted = generate_synthetic_recommender(
        logging_policy="popularity",
        popularity_strength=3.0,
        n_users=300,
        n_items=80,
        seed=9,
    )

    aligned_corr = np.corrcoef(aligned.popularity, aligned.propensity)[0, 1]
    distorted_corr = np.corrcoef(
        distorted.popularity, distorted.propensity
    )[0, 1]

    assert distorted_corr > aligned_corr


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_users": 0},
        {"n_items": 0},
        {"latent_dim": 0},
        {"min_propensity": 0.0},
        {"max_propensity": 1.0},
        {"min_propensity": 0.8, "max_propensity": 0.2},
    ],
)
def test_invalid_configuration_is_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        generate_synthetic_recommender(**kwargs)


def test_invalid_logging_policy_is_rejected() -> None:
    with pytest.raises(ValueError):
        generate_synthetic_recommender(
            logging_policy="unknown"  # type: ignore[arg-type]
        )
