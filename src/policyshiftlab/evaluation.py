"""Offline evaluation estimators for selectively observed data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from policyshiftlab.metrics import brier_score, effective_sample_size


@dataclass(frozen=True)
class BrierEvaluation:
    """Brier estimates for target and selectively observed populations."""

    target: float
    logged_naive: float
    logged_oracle_ipw: float
    logged_stratified: float
    ipw_effective_sample_size: float


def _validate_selection_inputs(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    selected: np.ndarray,
    propensity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    s = np.asarray(selected, dtype=bool)
    e = np.asarray(propensity, dtype=float)

    if not (y.ndim == p.ndim == s.ndim == e.ndim == 1):
        raise ValueError("all inputs must be one-dimensional")
    if not (y.shape == p.shape == s.shape == e.shape):
        raise ValueError("all inputs must have the same shape")
    if y.size == 0:
        raise ValueError("inputs must be non-empty")
    if not np.any(s):
        raise ValueError("selected must contain at least one observed example")
    if np.any((e <= 0.0) | (e > 1.0)):
        raise ValueError("propensity values must be in (0, 1]")

    # Reuse brier_score validation for labels and probabilities.
    brier_score(y, p)

    return y, p, s, e


def oracle_ipw_brier_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    selected: np.ndarray,
    propensity: np.ndarray,
) -> float:
    """Estimate target Brier score with the Horvitz-Thompson estimator.

    The target frame is represented by all input rows, while outcomes are used
    only for selected rows. With known inclusion propensities this estimator is
    unbiased for the finite-population mean under independent Bernoulli
    selection.
    """
    y, p, s, e = _validate_selection_inputs(
        y_true,
        y_prob,
        selected,
        propensity,
    )
    observed_loss = np.square(p[s] - y[s])
    return float(np.sum(observed_loss / e[s]) / y.size)


def propensity_stratified_brier_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    selected: np.ndarray,
    propensity: np.ndarray,
    *,
    n_strata: int = 5,
) -> float:
    """Estimate target Brier score using coarse propensity stratification.

    Target-population stratum masses are known from the candidate frame. The
    observed mean loss within each stratum is used as a coarse adjustment.
    """
    if n_strata < 2:
        raise ValueError("n_strata must be at least 2")

    y, p, s, e = _validate_selection_inputs(
        y_true,
        y_prob,
        selected,
        propensity,
    )
    losses = np.square(p - y)

    quantiles = np.linspace(0.0, 1.0, n_strata + 1)
    edges = np.quantile(e, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf

    estimate = 0.0

    for idx in range(n_strata):
        if idx == n_strata - 1:
            target_mask = (e >= edges[idx]) & (e <= edges[idx + 1])
        else:
            target_mask = (e >= edges[idx]) & (e < edges[idx + 1])

        target_mass = float(np.mean(target_mask))
        if target_mass == 0.0:
            continue

        observed_mask = target_mask & s
        if not np.any(observed_mask):
            raise ValueError(
                "propensity stratum has target mass but no observed examples"
            )

        estimate += target_mass * float(np.mean(losses[observed_mask]))

    return estimate


def evaluate_brier_under_selection(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    selected: np.ndarray,
    propensity: np.ndarray,
    *,
    n_strata: int = 5,
) -> BrierEvaluation:
    """Return target and selectively observed Brier-score estimates."""
    y, p, s, e = _validate_selection_inputs(
        y_true,
        y_prob,
        selected,
        propensity,
    )

    target = brier_score(y, p)
    logged_naive = brier_score(y[s], p[s])
    logged_oracle_ipw = oracle_ipw_brier_score(y, p, s, e)
    logged_stratified = propensity_stratified_brier_score(
        y,
        p,
        s,
        e,
        n_strata=n_strata,
    )

    return BrierEvaluation(
        target=target,
        logged_naive=logged_naive,
        logged_oracle_ipw=logged_oracle_ipw,
        logged_stratified=logged_stratified,
        ipw_effective_sample_size=effective_sample_size(1.0 / e[s]),
    )
