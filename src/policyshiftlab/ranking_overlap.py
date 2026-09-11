"""Overlap sweep for model-ranking stability."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from policyshiftlab.ranking import fixed_candidate_predictions
from policyshiftlab.ranking_monte_carlo import run_ranking_monte_carlo
from policyshiftlab.synthetic import generate_synthetic_recommender


@dataclass(frozen=True)
class RankingOverlapResult:
    """Ranking-stability summary for one overlap regime."""

    label: str
    min_propensity: float
    exposure_strength: float
    mean_propensity: float
    expected_selected: float
    expected_ipw_effective_sample_size: float
    logged_mean_spearman: float
    logged_mean_kendall: float
    logged_any_reversal_rate: float
    logged_exact_order_recovery_rate: float
    ipw_mean_spearman: float
    ipw_mean_kendall: float
    ipw_any_reversal_rate: float
    ipw_exact_order_recovery_rate: float


def expected_ipw_effective_sample_size(propensity: np.ndarray) -> float:
    """Deterministic expected-contribution approximation to IPW ESS."""
    e = np.asarray(propensity, dtype=float)

    if e.ndim != 1 or e.size == 0:
        raise ValueError("propensity must be a non-empty 1D array")
    if np.any((e <= 0.0) | (e > 1.0)):
        raise ValueError("propensity values must lie in (0, 1]")

    return float(e.size**2 / np.sum(1.0 / e))


def run_ranking_overlap_sweep(
    *,
    regimes: tuple[tuple[str, float, float], ...] = (
        ("strong", 0.20, 1.0),
        ("moderate", 0.10, 1.5),
        ("weak", 0.02, 3.0),
    ),
    n_users: int = 200,
    n_items: int = 80,
    n_replications: int = 300,
    seed: int = 0,
) -> list[RankingOverlapResult]:
    """Run ranking Monte Carlo across progressively weaker overlap.

    The synthetic generator is called with the same seed in every regime.
    Because only propensity-related parameters change, the target population,
    outcomes, and candidate predictions remain fixed across the sweep.
    """
    if not regimes:
        raise ValueError("at least one overlap regime is required")

    reference_outcome: np.ndarray | None = None
    reference_probability: np.ndarray | None = None
    reference_logging_score: np.ndarray | None = None
    results: list[RankingOverlapResult] = []

    for index, (label, min_propensity, exposure_strength) in enumerate(regimes):
        if not label:
            raise ValueError("regime labels must be non-empty")
        if not 0.0 < min_propensity < 1.0:
            raise ValueError("min_propensity must lie in (0, 1)")
        if exposure_strength <= 0.0:
            raise ValueError("exposure_strength must be positive")

        data = generate_synthetic_recommender(
            n_users=n_users,
            n_items=n_items,
            logging_policy="popularity",
            popularity_strength=3.0,
            exposure_strength=exposure_strength,
            min_propensity=min_propensity,
            max_propensity=0.95,
            seed=seed,
        )

        if reference_outcome is None:
            reference_outcome = data.outcome.copy()
            reference_probability = data.true_outcome_prob.copy()
            reference_logging_score = data.logging_score.copy()
        else:
            np.testing.assert_array_equal(data.outcome, reference_outcome)
            np.testing.assert_allclose(
                data.true_outcome_prob,
                reference_probability,
            )
            np.testing.assert_allclose(
                data.logging_score,
                reference_logging_score,
            )

        predictions = fixed_candidate_predictions(
            data.true_outcome_prob,
            data.logging_score,
        )
        summary = run_ranking_monte_carlo(
            data.outcome,
            predictions,
            data.propensity,
            n_replications=n_replications,
            seed=seed + 10_000 + index,
        )

        results.append(
            RankingOverlapResult(
                label=label,
                min_propensity=min_propensity,
                exposure_strength=exposure_strength,
                mean_propensity=float(np.mean(data.propensity)),
                expected_selected=float(np.sum(data.propensity)),
                expected_ipw_effective_sample_size=(
                    expected_ipw_effective_sample_size(data.propensity)
                ),
                logged_mean_spearman=summary.logged.mean_spearman,
                logged_mean_kendall=summary.logged.mean_kendall,
                logged_any_reversal_rate=summary.logged.any_reversal_rate,
                logged_exact_order_recovery_rate=(
                    summary.logged.exact_order_recovery_rate
                ),
                ipw_mean_spearman=summary.oracle_ipw.mean_spearman,
                ipw_mean_kendall=summary.oracle_ipw.mean_kendall,
                ipw_any_reversal_rate=(
                    summary.oracle_ipw.any_reversal_rate
                ),
                ipw_exact_order_recovery_rate=(
                    summary.oracle_ipw.exact_order_recovery_rate
                ),
            )
        )

    return results
