"""Overlap stress tests for selective-observation evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from policyshiftlab.monte_carlo import run_selection_monte_carlo
from policyshiftlab.synthetic import generate_synthetic_recommender


@dataclass(frozen=True)
class OverlapStressResult:
    """Summary of estimator behavior for one overlap configuration."""

    min_propensity: float
    exposure_strength: float
    mean_propensity: float
    min_observed_propensity: float
    oracle_ipw_bias: float
    oracle_ipw_rmse: float
    naive_bias: float
    naive_rmse: float
    stratified_bias: float
    stratified_rmse: float
    mean_ipw_effective_sample_size: float
    expected_selected: float


def _expected_ipw_effective_sample_size(propensity: np.ndarray) -> float:
    """Approximate expected ESS using expected Bernoulli contributions.

    This uses E[sum(S/e)] and E[sum((S/e)^2)] as a deterministic diagnostic.
    It is not the expectation of the random ESS itself.
    """
    e = np.asarray(propensity, dtype=float)
    numerator = float(e.size**2)
    denominator = float(np.sum(1.0 / e))
    return numerator / denominator


def run_overlap_stress_test(
    *,
    min_propensities: tuple[float, ...] = (0.20, 0.10, 0.05, 0.02),
    exposure_strengths: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0),
    n_users: int = 120,
    n_items: int = 60,
    n_replications: int = 300,
    seed: int = 0,
) -> list[OverlapStressResult]:
    """Evaluate estimators while progressively weakening overlap."""
    if len(min_propensities) != len(exposure_strengths):
        raise ValueError(
            "min_propensities and exposure_strengths must have equal length"
        )
    if not min_propensities:
        raise ValueError("at least one overlap configuration is required")

    results: list[OverlapStressResult] = []

    for index, (min_propensity, exposure_strength) in enumerate(
        zip(min_propensities, exposure_strengths, strict=True)
    ):
        if not 0.0 < min_propensity < 1.0:
            raise ValueError("min_propensities must lie in (0, 1)")
        if exposure_strength <= 0.0:
            raise ValueError("exposure_strengths must be positive")

        data = generate_synthetic_recommender(
            n_users=n_users,
            n_items=n_items,
            logging_policy="popularity",
            popularity_strength=3.0,
            exposure_strength=exposure_strength,
            min_propensity=min_propensity,
            max_propensity=0.95,
            seed=seed + index,
        )

        summaries = run_selection_monte_carlo(
            data.outcome,
            data.true_outcome_prob,
            data.propensity,
            n_replications=n_replications,
            seed=seed + 10_000 + index,
        )

        results.append(
            OverlapStressResult(
                min_propensity=min_propensity,
                exposure_strength=exposure_strength,
                mean_propensity=float(np.mean(data.propensity)),
                min_observed_propensity=float(np.min(data.propensity)),
                oracle_ipw_bias=summaries["oracle_ipw"].bias,
                oracle_ipw_rmse=summaries["oracle_ipw"].rmse,
                naive_bias=summaries["naive"].bias,
                naive_rmse=summaries["naive"].rmse,
                stratified_bias=summaries["stratified"].bias,
                stratified_rmse=summaries["stratified"].rmse,
                mean_ipw_effective_sample_size=(
                    _expected_ipw_effective_sample_size(data.propensity)
                ),
                expected_selected=float(np.sum(data.propensity)),
            )
        )

    return results
