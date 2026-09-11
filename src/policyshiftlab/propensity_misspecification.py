"""Propensity misspecification diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from policyshiftlab.metrics import brier_score
from policyshiftlab.ranking import compare_model_rankings, fixed_candidate_predictions
from policyshiftlab.synthetic import generate_synthetic_recommender


@dataclass(frozen=True)
class PropensityScenarioResult:
    """Evaluation summary under one propensity specification."""

    label: str
    brier_estimate: float
    brier_error: float
    mean_spearman: float
    mean_kendall: float
    any_reversal_rate: float
    exact_order_recovery_rate: float


@dataclass(frozen=True)
class PropensityMisspecificationResult:
    """Target and corrected-evaluation summaries."""

    target_brier: float
    naive_logged_brier: float
    oracle: PropensityScenarioResult
    mild_misspecified: PropensityScenarioResult
    severe_misspecified: PropensityScenarioResult


def make_misspecified_propensities(
    true_propensity: np.ndarray,
    logging_score: np.ndarray,
) -> dict[str, np.ndarray]:
    """Construct fixed diagnostic propensity misspecification scenarios.

    ``mild`` shrinks true propensities toward their marginal mean.
    ``severe`` reverses much of the score-dependent ordering before clipping.
    These are controlled perturbations, not fitted propensity estimators.
    """
    e = np.asarray(true_propensity, dtype=float)
    score = np.asarray(logging_score, dtype=float)

    if e.ndim != 1 or score.ndim != 1 or e.shape != score.shape:
        raise ValueError(
            "true_propensity and logging_score must be matching 1D arrays"
        )
    if np.any((e <= 0.0) | (e > 1.0)):
        raise ValueError("true_propensity values must lie in (0, 1]")

    mean_e = float(np.mean(e))
    mild = 0.80 * e + 0.20 * mean_e

    centered_score = score - float(np.mean(score))
    severe = 1.0 / (1.0 + np.exp(centered_score))
    severe = 0.10 + 0.80 * severe

    return {
        "oracle": e.copy(),
        "mild": np.clip(mild, 0.02, 0.98),
        "severe": np.clip(severe, 0.02, 0.98),
    }


def _ht_brier_with_supplied_propensity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    selected: np.ndarray,
    supplied_propensity: np.ndarray,
) -> float:
    """Horvitz-Thompson-style Brier estimate using supplied propensities."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    s = np.asarray(selected, dtype=bool)
    e_hat = np.asarray(supplied_propensity, dtype=float)

    if e_hat.shape != y.shape:
        raise ValueError("supplied_propensity must match y_true")
    if np.any((e_hat <= 0.0) | (e_hat > 1.0)):
        raise ValueError("supplied_propensity values must lie in (0, 1]")

    losses = np.square(p[s] - y[s])
    return float(np.sum(losses / e_hat[s]) / y.size)


def _ranking_summary_under_misspecification(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    true_propensity: np.ndarray,
    supplied_propensity: np.ndarray,
    *,
    n_replications: int,
    seed: int,
) -> tuple[float, float, float, float]:
    """Redraw from true logging policy, evaluate with supplied propensities."""
    y = np.asarray(y_true, dtype=float)
    e_true = np.asarray(true_propensity, dtype=float)
    e_hat = np.asarray(supplied_propensity, dtype=float)

    if e_true.shape != y.shape or e_hat.shape != y.shape:
        raise ValueError("propensity arrays must match y_true")
    if n_replications < 2:
        raise ValueError("n_replications must be at least 2")

    rng = np.random.default_rng(seed)
    spearman = np.empty(n_replications, dtype=float)
    kendall = np.empty(n_replications, dtype=float)
    reversal = np.empty(n_replications, dtype=bool)
    exact = np.empty(n_replications, dtype=bool)

    for replication in range(n_replications):
        selected = rng.random(y.size) < e_true
        if not np.any(selected):
            selected[rng.integers(0, y.size)] = True

        comparison = compare_model_rankings(
            y,
            predictions,
            selected,
            e_hat,
        )

        spearman[replication] = comparison.oracle_ipw_spearman
        kendall[replication] = comparison.oracle_ipw_kendall
        reversal[replication] = bool(comparison.oracle_ipw_reversals)
        exact[replication] = (
            comparison.oracle_ipw_order == comparison.target_order
        )

    return (
        float(np.nanmean(spearman)),
        float(np.nanmean(kendall)),
        float(np.mean(reversal)),
        float(np.mean(exact)),
    )


def run_propensity_misspecification_experiment(
    *,
    n_users: int = 200,
    n_items: int = 80,
    n_replications: int = 300,
    seed: int = 0,
) -> PropensityMisspecificationResult:
    """Compare oracle and misspecified weighting under one fixed logging policy."""
    data = generate_synthetic_recommender(
        n_users=n_users,
        n_items=n_items,
        logging_policy="popularity",
        popularity_strength=3.0,
        exposure_strength=2.0,
        min_propensity=0.05,
        max_propensity=0.95,
        seed=seed,
    )

    predictions = fixed_candidate_predictions(
        data.true_outcome_prob,
        data.logging_score,
    )
    propensity_scenarios = make_misspecified_propensities(
        data.propensity,
        data.logging_score,
    )

    target_brier = brier_score(
        data.outcome,
        data.true_outcome_prob,
    )
    naive_logged_brier = brier_score(
        data.outcome[data.selected],
        data.true_outcome_prob[data.selected],
    )

    scenario_results: dict[str, PropensityScenarioResult] = {}

    for index, (label, supplied_propensity) in enumerate(
        propensity_scenarios.items()
    ):
        estimate = _ht_brier_with_supplied_propensity(
            data.outcome,
            data.true_outcome_prob,
            data.selected,
            supplied_propensity,
        )

        (
            mean_spearman,
            mean_kendall,
            any_reversal_rate,
            exact_order_recovery_rate,
        ) = _ranking_summary_under_misspecification(
            data.outcome,
            predictions,
            data.propensity,
            supplied_propensity,
            n_replications=n_replications,
            seed=seed + 10_000 + index,
        )

        scenario_results[label] = PropensityScenarioResult(
            label=label,
            brier_estimate=estimate,
            brier_error=estimate - target_brier,
            mean_spearman=mean_spearman,
            mean_kendall=mean_kendall,
            any_reversal_rate=any_reversal_rate,
            exact_order_recovery_rate=exact_order_recovery_rate,
        )

    return PropensityMisspecificationResult(
        target_brier=target_brier,
        naive_logged_brier=naive_logged_brier,
        oracle=scenario_results["oracle"],
        mild_misspecified=scenario_results["mild"],
        severe_misspecified=scenario_results["severe"],
    )
