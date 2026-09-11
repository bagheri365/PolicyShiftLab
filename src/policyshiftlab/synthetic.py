"""Synthetic recommender data-generating process.

The simulator creates a target population of user-item pairs with known outcome
probabilities and then samples logged observations through a configurable
exposure policy. This gives PolicyShiftLab access to oracle propensities for
controlled evaluation experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

LoggingPolicy = Literal["aligned", "popularity"]


@dataclass(frozen=True)
class SyntheticRecommenderData:
    """Synthetic user-item population and logged-selection variables."""

    user_id: np.ndarray
    item_id: np.ndarray
    true_outcome_prob: np.ndarray
    outcome: np.ndarray
    propensity: np.ndarray
    selected: np.ndarray
    popularity: np.ndarray
    logging_score: np.ndarray

    @property
    def n_pairs(self) -> int:
        """Return number of user-item pairs in the target population."""
        return int(self.outcome.size)

    @property
    def n_selected(self) -> int:
        """Return number of pairs observed by the logging policy."""
        return int(np.sum(self.selected))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    positive = values >= 0

    result = np.empty_like(values)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))

    negative_values = values[~positive]
    exp_values = np.exp(negative_values)
    result[~positive] = exp_values / (1.0 + exp_values)

    return result


def generate_synthetic_recommender(
    *,
    n_users: int = 200,
    n_items: int = 100,
    latent_dim: int = 4,
    logging_policy: LoggingPolicy = "aligned",
    exposure_strength: float = 1.5,
    popularity_strength: float = 1.0,
    min_propensity: float = 0.05,
    max_propensity: float = 0.95,
    seed: int = 0,
) -> SyntheticRecommenderData:
    """Generate a synthetic recommender population and logged exposure sample."""
    if n_users <= 0 or n_items <= 0 or latent_dim <= 0:
        raise ValueError("n_users, n_items, and latent_dim must be positive")
    if logging_policy not in {"aligned", "popularity"}:
        raise ValueError("logging_policy must be 'aligned' or 'popularity'")
    if not 0.0 < min_propensity < max_propensity < 1.0:
        raise ValueError(
            "propensity bounds must satisfy 0 < min_propensity < "
            "max_propensity < 1"
        )

    rng = np.random.default_rng(seed)

    user_factors = rng.normal(0.0, 1.0, size=(n_users, latent_dim))
    item_factors = rng.normal(0.0, 1.0, size=(n_items, latent_dim))
    user_bias = rng.normal(0.0, 0.35, size=n_users)
    item_bias = rng.normal(0.0, 0.35, size=n_items)

    user_id = np.repeat(np.arange(n_users), n_items)
    item_id = np.tile(np.arange(n_items), n_users)

    outcome_logits = (
        np.sum(user_factors[user_id] * item_factors[item_id], axis=1)
        / np.sqrt(latent_dim)
        + user_bias[user_id]
        + item_bias[item_id]
    )
    true_outcome_prob = _sigmoid(outcome_logits)
    outcome = rng.binomial(1, true_outcome_prob).astype(int)

    item_popularity_logits = 0.8 * item_bias + rng.normal(0.0, 0.5, size=n_items)
    item_popularity = _sigmoid(item_popularity_logits)
    popularity = item_popularity[item_id]

    old_user_factors = user_factors + rng.normal(
        0.0, 0.35, size=user_factors.shape
    )
    old_item_factors = item_factors + rng.normal(
        0.0, 0.35, size=item_factors.shape
    )

    logging_score = (
        np.sum(
            old_user_factors[user_id] * old_item_factors[item_id],
            axis=1,
        )
        / np.sqrt(latent_dim)
        + rng.normal(0.0, 0.35, size=user_id.size)
    )

    if logging_policy == "popularity":
        centered_popularity = popularity - float(np.mean(popularity))
        logging_score = (
            logging_score + popularity_strength * centered_popularity
        )

    raw_propensity = _sigmoid(exposure_strength * logging_score)
    propensity = np.clip(raw_propensity, min_propensity, max_propensity)
    selected = rng.binomial(1, propensity).astype(bool)

    return SyntheticRecommenderData(
        user_id=user_id,
        item_id=item_id,
        true_outcome_prob=true_outcome_prob,
        outcome=outcome,
        propensity=propensity,
        selected=selected,
        popularity=popularity,
        logging_score=logging_score,
    )
