"""Monte Carlo ranking stability under repeated selective observation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from policyshiftlab.ranking import compare_model_rankings


@dataclass(frozen=True)
class RankingMonteCarloSummary:
    """Repeated-selection summary for one evaluation method."""

    method: str
    mean_spearman: float
    mean_kendall: float
    median_spearman: float
    median_kendall: float
    any_reversal_rate: float
    mean_reversal_count: float
    exact_order_recovery_rate: float
    n_replications: int


@dataclass(frozen=True)
class RankingMonteCarloResult:
    """Naive and oracle-IPW ranking stability summaries."""

    logged: RankingMonteCarloSummary
    oracle_ipw: RankingMonteCarloSummary


def _summarize(
    *,
    method: str,
    spearman: np.ndarray,
    kendall: np.ndarray,
    reversal_count: np.ndarray,
    exact_order: np.ndarray,
) -> RankingMonteCarloSummary:
    return RankingMonteCarloSummary(
        method=method,
        mean_spearman=float(np.nanmean(spearman)),
        mean_kendall=float(np.nanmean(kendall)),
        median_spearman=float(np.nanmedian(spearman)),
        median_kendall=float(np.nanmedian(kendall)),
        any_reversal_rate=float(np.mean(reversal_count > 0)),
        mean_reversal_count=float(np.mean(reversal_count)),
        exact_order_recovery_rate=float(np.mean(exact_order)),
        n_replications=int(spearman.size),
    )


def run_ranking_monte_carlo(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    propensity: np.ndarray,
    *,
    n_replications: int = 500,
    seed: int = 0,
) -> RankingMonteCarloResult:
    """Redraw selection and summarize ranking stability.

    The target population, outcomes, candidate predictions, and propensities
    are held fixed. Only Bernoulli selection is redrawn. This isolates
    finite-sample model-selection instability induced by the logging policy.
    """
    y = np.asarray(y_true, dtype=float)
    e = np.asarray(propensity, dtype=float)

    if y.ndim != 1 or y.size == 0:
        raise ValueError("y_true must be a non-empty 1D array")
    if e.ndim != 1 or e.shape != y.shape:
        raise ValueError("propensity must match y_true")
    if np.any((e <= 0.0) | (e > 1.0)):
        raise ValueError("propensity values must lie in (0, 1]")
    if n_replications < 2:
        raise ValueError("n_replications must be at least 2")

    rng = np.random.default_rng(seed)

    logged_spearman = np.empty(n_replications, dtype=float)
    logged_kendall = np.empty(n_replications, dtype=float)
    logged_reversals = np.empty(n_replications, dtype=int)
    logged_exact = np.empty(n_replications, dtype=bool)

    ipw_spearman = np.empty(n_replications, dtype=float)
    ipw_kendall = np.empty(n_replications, dtype=float)
    ipw_reversals = np.empty(n_replications, dtype=int)
    ipw_exact = np.empty(n_replications, dtype=bool)

    for replication in range(n_replications):
        selected = rng.random(y.size) < e

        # Vanishingly unlikely in the intended experiments, but keep the
        # routine well-defined for tiny test fixtures or extreme propensities.
        if not np.any(selected):
            selected[rng.integers(0, y.size)] = True

        comparison = compare_model_rankings(
            y,
            predictions,
            selected,
            e,
        )

        logged_spearman[replication] = comparison.logged_spearman
        logged_kendall[replication] = comparison.logged_kendall
        logged_reversals[replication] = len(comparison.logged_reversals)
        logged_exact[replication] = (
            comparison.logged_order == comparison.target_order
        )

        ipw_spearman[replication] = comparison.oracle_ipw_spearman
        ipw_kendall[replication] = comparison.oracle_ipw_kendall
        ipw_reversals[replication] = len(
            comparison.oracle_ipw_reversals
        )
        ipw_exact[replication] = (
            comparison.oracle_ipw_order == comparison.target_order
        )

    return RankingMonteCarloResult(
        logged=_summarize(
            method="logged",
            spearman=logged_spearman,
            kendall=logged_kendall,
            reversal_count=logged_reversals,
            exact_order=logged_exact,
        ),
        oracle_ipw=_summarize(
            method="oracle_ipw",
            spearman=ipw_spearman,
            kendall=ipw_kendall,
            reversal_count=ipw_reversals,
            exact_order=ipw_exact,
        ),
    )
