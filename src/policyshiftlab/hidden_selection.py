"""Failure case with a hidden selection driver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from policyshiftlab.metrics import brier_score
from policyshiftlab.synthetic import generate_synthetic_recommender


@dataclass(frozen=True)
class HiddenSelectionData:
    """Finite target population with an unobserved selection driver."""

    outcome: np.ndarray
    observed_prediction: np.ndarray
    hidden_variable: np.ndarray
    full_propensity: np.ndarray
    observed_only_propensity: np.ndarray


@dataclass(frozen=True)
class EstimatorSummary:
    """Repeated-selection estimator summary."""

    estimator: str
    target: float
    mean_estimate: float
    bias: float
    variance: float
    rmse: float
    n_replications: int


@dataclass(frozen=True)
class HiddenSelectionResult:
    """Comparison of naive, full-IPW, and observed-only IPW evaluation."""

    target_brier: float
    naive: EstimatorSummary
    full_ipw: EstimatorSummary
    observed_only_ipw: EstimatorSummary


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    positive = values >= 0.0
    result = np.empty_like(values, dtype=float)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def _logit(probability: np.ndarray) -> np.ndarray:
    p = np.asarray(probability, dtype=float)
    clipped = np.clip(p, 1e-8, 1.0 - 1e-8)
    return np.log(clipped / (1.0 - clipped))


def generate_hidden_selection_data(
    *,
    n_users: int = 200,
    n_items: int = 80,
    outcome_hidden_strength: float = 1.0,
    selection_hidden_strength: float = 1.5,
    exposure_strength: float = 1.5,
    min_propensity: float = 0.05,
    max_propensity: float = 0.95,
    seed: int = 0,
) -> HiddenSelectionData:
    """Generate a population violating Y independent of S given observed X.

    A binary hidden variable ``Z`` affects both the outcome probability and the
    logging propensity. The observed-only propensity is not an arbitrary bad
    model: because Z is independent of the observed synthetic covariates here,
    it is the exact marginal propensity P(S=1 | X), obtained by integrating Z
    out. It is nevertheless insufficient for identifying the target Brier
    score because Y and S remain dependent conditional on observed X.
    """
    if outcome_hidden_strength <= 0.0:
        raise ValueError("outcome_hidden_strength must be positive")
    if selection_hidden_strength <= 0.0:
        raise ValueError("selection_hidden_strength must be positive")
    if exposure_strength <= 0.0:
        raise ValueError("exposure_strength must be positive")
    if not 0.0 < min_propensity < max_propensity <= 1.0:
        raise ValueError(
            "propensity bounds must satisfy 0 < min < max <= 1"
        )

    base = generate_synthetic_recommender(
        n_users=n_users,
        n_items=n_items,
        logging_policy="popularity",
        popularity_strength=3.0,
        exposure_strength=1.0,
        min_propensity=min_propensity,
        max_propensity=max_propensity,
        seed=seed,
    )

    rng = np.random.default_rng(seed + 91_337)
    hidden = rng.choice(np.array([-1.0, 1.0]), size=base.n_pairs)

    base_logit = _logit(base.true_outcome_prob)
    full_outcome_prob = _sigmoid(
        base_logit + outcome_hidden_strength * hidden
    )
    outcome = rng.binomial(1, full_outcome_prob)

    observed_prediction = base.true_outcome_prob.copy()

    base_selection_score = exposure_strength * base.logging_score
    full_propensity = _sigmoid(
        base_selection_score + selection_hidden_strength * hidden
    )
    full_propensity = np.clip(
        full_propensity,
        min_propensity,
        max_propensity,
    )

    plus = np.clip(
        _sigmoid(base_selection_score + selection_hidden_strength),
        min_propensity,
        max_propensity,
    )
    minus = np.clip(
        _sigmoid(base_selection_score - selection_hidden_strength),
        min_propensity,
        max_propensity,
    )
    observed_only_propensity = 0.5 * (plus + minus)

    return HiddenSelectionData(
        outcome=outcome,
        observed_prediction=observed_prediction,
        hidden_variable=hidden,
        full_propensity=full_propensity,
        observed_only_propensity=observed_only_propensity,
    )


def _ht_brier(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    selected: np.ndarray,
    propensity: np.ndarray,
) -> float:
    losses = np.square(y_prob[selected] - y_true[selected])
    return float(np.sum(losses / propensity[selected]) / y_true.size)


def _summarize(
    estimator: str,
    estimates: np.ndarray,
    target: float,
) -> EstimatorSummary:
    error = estimates - target
    return EstimatorSummary(
        estimator=estimator,
        target=target,
        mean_estimate=float(np.mean(estimates)),
        bias=float(np.mean(error)),
        variance=float(np.var(estimates, ddof=1)),
        rmse=float(np.sqrt(np.mean(np.square(error)))),
        n_replications=int(estimates.size),
    )


def run_hidden_selection_monte_carlo(
    data: HiddenSelectionData,
    *,
    n_replications: int = 500,
    seed: int = 0,
) -> HiddenSelectionResult:
    """Redraw selection from the full propensity and compare estimators."""
    if n_replications < 2:
        raise ValueError("n_replications must be at least 2")

    y = np.asarray(data.outcome, dtype=float)
    p = np.asarray(data.observed_prediction, dtype=float)
    full_e = np.asarray(data.full_propensity, dtype=float)
    observed_e = np.asarray(data.observed_only_propensity, dtype=float)

    if not (
        y.shape == p.shape == full_e.shape == observed_e.shape
        and y.ndim == 1
    ):
        raise ValueError("all HiddenSelectionData arrays must match")
    if np.any((full_e <= 0.0) | (full_e > 1.0)):
        raise ValueError("full_propensity values must lie in (0, 1]")
    if np.any((observed_e <= 0.0) | (observed_e > 1.0)):
        raise ValueError(
            "observed_only_propensity values must lie in (0, 1]"
        )

    target = brier_score(y, p)
    rng = np.random.default_rng(seed)

    naive = np.empty(n_replications, dtype=float)
    full_ipw = np.empty(n_replications, dtype=float)
    observed_ipw = np.empty(n_replications, dtype=float)

    for replication in range(n_replications):
        selected = rng.random(y.size) < full_e
        if not np.any(selected):
            selected[rng.integers(0, y.size)] = True

        naive[replication] = brier_score(y[selected], p[selected])
        full_ipw[replication] = _ht_brier(y, p, selected, full_e)
        observed_ipw[replication] = _ht_brier(
            y,
            p,
            selected,
            observed_e,
        )

    return HiddenSelectionResult(
        target_brier=target,
        naive=_summarize("naive", naive, target),
        full_ipw=_summarize("full_ipw", full_ipw, target),
        observed_only_ipw=_summarize(
            "observed_only_ipw",
            observed_ipw,
            target,
        ),
    )
