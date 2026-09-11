"""Diagnostics for the supplied Coat propensity matrix."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CoatPropensityAudit:
    n_users: int
    n_items: int
    min_propensity: float
    median_propensity: float
    max_propensity: float
    mean_propensity: float
    q01_propensity: float
    q05_propensity: float
    q95_propensity: float
    q99_propensity: float
    row_sum_min: float
    row_sum_median: float
    row_sum_max: float
    expected_row_observations: float
    train_observed_mean_propensity: float
    train_unobserved_mean_propensity: float
    randomized_observed_mean_propensity: float
    train_observed_weight_min: float
    train_observed_weight_median: float
    train_observed_weight_max: float
    train_observed_weight_ess: float


def load_coat_propensities(path: str | Path) -> np.ndarray:
    """Load the learned Coat propensity matrix."""
    prop = np.loadtxt(Path(path), dtype=float)

    if prop.ndim != 2:
        raise ValueError("propensity matrix must be two-dimensional")
    if prop.size == 0:
        raise ValueError("propensity matrix must be non-empty")
    if not np.all(np.isfinite(prop)):
        raise ValueError("propensities must be finite")
    if np.any(prop <= 0.0) or np.any(prop > 1.0):
        raise ValueError("propensities must lie in (0, 1]")

    return prop


def _ess(weights: np.ndarray) -> float:
    denom = float(np.sum(np.square(weights)))
    if denom == 0.0:
        return 0.0
    total = float(np.sum(weights))
    return total * total / denom


def audit_coat_propensities(
    train: np.ndarray,
    randomized: np.ndarray,
    propensities: np.ndarray,
) -> CoatPropensityAudit:
    """Summarize support and weight behavior of supplied Coat propensities."""
    train = np.asarray(train)
    randomized = np.asarray(randomized)
    prop = np.asarray(propensities, dtype=float)

    if train.ndim != 2 or randomized.ndim != 2 or prop.ndim != 2:
        raise ValueError("train, randomized, and propensities must be 2D")
    if train.shape != randomized.shape or train.shape != prop.shape:
        raise ValueError("all matrices must have identical shapes")
    if not np.all(np.isfinite(prop)):
        raise ValueError("propensities must be finite")
    if np.any(prop <= 0.0) or np.any(prop > 1.0):
        raise ValueError("propensities must lie in (0, 1]")

    train_mask = train > 0
    randomized_mask = randomized > 0
    if not np.any(train_mask):
        raise ValueError("train matrix has no observed ratings")

    train_prop = prop[train_mask]
    unobserved_prop = prop[~train_mask]
    randomized_prop = prop[randomized_mask]
    weights = 1.0 / train_prop
    row_sums = np.sum(prop, axis=1)
    quantiles = np.quantile(prop, [0.01, 0.05, 0.5, 0.95, 0.99])

    n_users, n_items = prop.shape

    return CoatPropensityAudit(
        n_users=n_users,
        n_items=n_items,
        min_propensity=float(np.min(prop)),
        median_propensity=float(quantiles[2]),
        max_propensity=float(np.max(prop)),
        mean_propensity=float(np.mean(prop)),
        q01_propensity=float(quantiles[0]),
        q05_propensity=float(quantiles[1]),
        q95_propensity=float(quantiles[3]),
        q99_propensity=float(quantiles[4]),
        row_sum_min=float(np.min(row_sums)),
        row_sum_median=float(np.median(row_sums)),
        row_sum_max=float(np.max(row_sums)),
        expected_row_observations=float(np.mean(row_sums)),
        train_observed_mean_propensity=float(np.mean(train_prop)),
        train_unobserved_mean_propensity=float(np.mean(unobserved_prop)),
        randomized_observed_mean_propensity=float(np.mean(randomized_prop)),
        train_observed_weight_min=float(np.min(weights)),
        train_observed_weight_median=float(np.median(weights)),
        train_observed_weight_max=float(np.max(weights)),
        train_observed_weight_ess=_ess(weights),
    )
