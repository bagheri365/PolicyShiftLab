"""Monte Carlo evaluation for selective-observation estimators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from policyshiftlab.evaluation import (
    oracle_ipw_brier_score,
    propensity_stratified_brier_score,
)


@dataclass(frozen=True)
class MonteCarloSummary:
    """Summary statistics for one estimator across repeated selection draws."""

    estimator: str
    target: float
    mean_estimate: float
    bias: float
    variance: float
    rmse: float
    coverage_95: float
    mean_interval_width: float
    n_replications: int


def _normal_interval(
    estimate: float,
    standard_error: float,
    *,
    z_value: float = 1.959963984540054,
) -> tuple[float, float]:
    half_width = z_value * standard_error
    return estimate - half_width, estimate + half_width


def _naive_logged_brier(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    selected: np.ndarray,
) -> float:
    losses = np.square(y_prob[selected] - y_true[selected])
    return float(np.mean(losses))


def _naive_logged_standard_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    selected: np.ndarray,
) -> float:
    losses = np.square(y_prob[selected] - y_true[selected])
    if losses.size < 2:
        return float("nan")
    return float(np.std(losses, ddof=1) / np.sqrt(losses.size))


def _oracle_ipw_standard_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    selected: np.ndarray,
    propensity: np.ndarray,
) -> float:
    """Approximate HT standard error under independent Bernoulli sampling."""
    losses = np.square(y_prob - y_true)
    terms = (
        selected.astype(float)
        * (1.0 - propensity)
        * np.square(losses / propensity)
    )
    variance = float(np.sum(terms) / (y_true.size**2))
    return float(np.sqrt(max(variance, 0.0)))


def run_selection_monte_carlo(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    propensity: np.ndarray,
    *,
    n_replications: int = 500,
    n_strata: int = 5,
    seed: int = 0,
) -> dict[str, MonteCarloSummary]:
    """Evaluate estimators across repeated draws from a fixed logging policy.

    The finite target population, outcomes, predictions, and propensities remain
    fixed. Only the Bernoulli selection indicators are redrawn. This isolates
    estimator behavior under the logging mechanism.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    e = np.asarray(propensity, dtype=float)

    if not (y.ndim == p.ndim == e.ndim == 1):
        raise ValueError("all inputs must be one-dimensional")
    if not (y.shape == p.shape == e.shape):
        raise ValueError("all inputs must have the same shape")
    if y.size == 0:
        raise ValueError("inputs must be non-empty")
    if n_replications < 2:
        raise ValueError("n_replications must be at least 2")
    if np.any((e <= 0.0) | (e > 1.0)):
        raise ValueError("propensity values must be in (0, 1]")

    target = float(np.mean(np.square(p - y)))
    rng = np.random.default_rng(seed)

    estimates: dict[str, list[float]] = {
        "naive": [],
        "oracle_ipw": [],
        "stratified": [],
    }
    intervals: dict[str, list[tuple[float, float]]] = {
        "naive": [],
        "oracle_ipw": [],
        "stratified": [],
    }

    for _ in range(n_replications):
        selected = rng.random(y.size) < e
        if not np.any(selected):
            continue

        naive = _naive_logged_brier(y, p, selected)
        ipw = oracle_ipw_brier_score(y, p, selected, e)

        try:
            stratified = propensity_stratified_brier_score(
                y,
                p,
                selected,
                e,
                n_strata=n_strata,
            )
        except ValueError:
            continue

        estimates["naive"].append(naive)
        estimates["oracle_ipw"].append(ipw)
        estimates["stratified"].append(stratified)

        naive_se = _naive_logged_standard_error(y, p, selected)
        ipw_se = _oracle_ipw_standard_error(y, p, selected, e)

        intervals["naive"].append(_normal_interval(naive, naive_se))
        intervals["oracle_ipw"].append(_normal_interval(ipw, ipw_se))
        intervals["stratified"].append((float("nan"), float("nan")))

    summaries: dict[str, MonteCarloSummary] = {}

    for name, values in estimates.items():
        arr = np.asarray(values, dtype=float)
        if arr.size < 2:
            raise RuntimeError("too few valid Monte Carlo replications")

        bias = float(np.mean(arr) - target)
        variance = float(np.var(arr, ddof=1))
        rmse = float(np.sqrt(np.mean(np.square(arr - target))))

        bounds = np.asarray(intervals[name], dtype=float)
        finite = np.isfinite(bounds).all(axis=1)
        if np.any(finite):
            coverage = float(
                np.mean(
                    (bounds[finite, 0] <= target)
                    & (target <= bounds[finite, 1])
                )
            )
            mean_width = float(
                np.mean(bounds[finite, 1] - bounds[finite, 0])
            )
        else:
            coverage = float("nan")
            mean_width = float("nan")

        summaries[name] = MonteCarloSummary(
            estimator=name,
            target=target,
            mean_estimate=float(np.mean(arr)),
            bias=bias,
            variance=variance,
            rmse=rmse,
            coverage_95=coverage,
            mean_interval_width=mean_width,
            n_replications=int(arr.size),
        )

    return summaries
