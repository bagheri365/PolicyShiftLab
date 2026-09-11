"""Repeated leakage-safe Yahoo! R3 holdouts and randomized uncertainty."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from policyshiftlab.yahoo_r3_empirical import (
    fit_yahoo_probability_models,
    predict_yahoo_probability_models,
    restrict_logged_to_randomized_users,
    run_yahoo_r3_heldout_benchmark,
    split_logged_triplets_by_user,
)


@dataclass(frozen=True)
class YahooRepeatedMethodSummary:
    method: str
    mean_signed_brier_gap: float
    mean_absolute_brier_gap: float
    rmse_brier_gap: float
    mean_spearman: float
    median_spearman: float
    mean_kendall: float
    median_kendall: float
    any_reversal_rate: float
    mean_reversal_count: float
    exact_order_recovery_rate: float
    n_splits: int


@dataclass(frozen=True)
class YahooPairReversalSummary:
    model_a: str
    model_b: str
    reversal_rate: float
    n_splits: int


@dataclass(frozen=True)
class YahooRandomizedPairBootstrap:
    model_a: str
    model_b: str
    observed_difference: float
    lower_95: float
    upper_95: float
    probability_a_better: float
    n_bootstrap: int
    seed: int


@dataclass(frozen=True)
class YahooRepeatedHoldoutResult:
    n_splits: int
    first_seed: int
    holdout_fraction: float
    method_summary: YahooRepeatedMethodSummary
    pair_reversal: YahooPairReversalSummary


def run_repeated_yahoo_r3_holdouts(
    train: np.ndarray,
    randomized: np.ndarray,
    *,
    n_splits: int = 200,
    first_seed: int = 0,
    holdout_fraction: float = 1.0 / 3.0,
    positive_threshold: int = 4,
    prior_strength: float = 10.0,
    pair: tuple[str, str] = ("user_smoothed", "user_item_blend"),
) -> YahooRepeatedHoldoutResult:
    """Repeat the leakage-safe Yahoo! R3 split over deterministic seeds.

    This measures sensitivity to the logged fit/evaluation split conditional on
    the observed Yahoo! R3 data. It is not replication over new datasets or new
    randomized-test samples.
    """
    if n_splits <= 0:
        raise ValueError("n_splits must be positive")
    if first_seed < 0:
        raise ValueError("first_seed must be non-negative")

    gaps: list[float] = []
    spearman: list[float] = []
    kendall: list[float] = []
    reversal_counts: list[int] = []
    exact_recovery: list[float] = []
    pair_reversed: list[float] = []

    model_a, model_b = pair

    for seed in range(first_seed, first_seed + n_splits):
        result = run_yahoo_r3_heldout_benchmark(
            train,
            randomized,
            positive_threshold=positive_threshold,
            prior_strength=prior_strength,
            holdout_fraction=holdout_fraction,
            seed=seed,
        )

        randomized_scores = {
            row.model: row.randomized_brier
            for row in result.scores
        }
        logged_scores = {
            row.model: row.heldout_logged_brier
            for row in result.scores
        }

        if model_a not in randomized_scores or model_b not in randomized_scores:
            raise ValueError("requested pair is not present in candidate models")

        gaps.extend(
            logged_scores[name] - randomized_scores[name]
            for name in randomized_scores
        )

        ranking = next(
            row
            for row in result.rankings
            if row.method == "heldout_logged_naive"
        )
        spearman.append(ranking.spearman_vs_randomized)
        kendall.append(ranking.kendall_vs_randomized)

        count = len(ranking.reversals_vs_randomized)
        reversal_counts.append(count)
        exact_recovery.append(float(count == 0))

        target_diff = randomized_scores[model_a] - randomized_scores[model_b]
        logged_diff = logged_scores[model_a] - logged_scores[model_b]
        pair_reversed.append(float(target_diff * logged_diff < 0.0))

    gap_arr = np.asarray(gaps, dtype=float)
    sp_arr = np.asarray(spearman, dtype=float)
    tau_arr = np.asarray(kendall, dtype=float)
    rev_arr = np.asarray(reversal_counts, dtype=float)
    exact_arr = np.asarray(exact_recovery, dtype=float)
    pair_arr = np.asarray(pair_reversed, dtype=float)

    return YahooRepeatedHoldoutResult(
        n_splits=n_splits,
        first_seed=first_seed,
        holdout_fraction=holdout_fraction,
        method_summary=YahooRepeatedMethodSummary(
            method="heldout_logged_naive",
            mean_signed_brier_gap=float(np.mean(gap_arr)),
            mean_absolute_brier_gap=float(np.mean(np.abs(gap_arr))),
            rmse_brier_gap=float(np.sqrt(np.mean(np.square(gap_arr)))),
            mean_spearman=float(np.nanmean(sp_arr)),
            median_spearman=float(np.nanmedian(sp_arr)),
            mean_kendall=float(np.nanmean(tau_arr)),
            median_kendall=float(np.nanmedian(tau_arr)),
            any_reversal_rate=float(np.mean(rev_arr > 0)),
            mean_reversal_count=float(np.mean(rev_arr)),
            exact_order_recovery_rate=float(np.mean(exact_arr)),
            n_splits=n_splits,
        ),
        pair_reversal=YahooPairReversalSummary(
            model_a=model_a,
            model_b=model_b,
            reversal_rate=float(np.mean(pair_arr)),
            n_splits=n_splits,
        ),
    )


def bootstrap_yahoo_randomized_pair_difference(
    train: np.ndarray,
    randomized: np.ndarray,
    *,
    fit_seed: int = 2026,
    holdout_fraction: float = 1.0 / 3.0,
    positive_threshold: int = 4,
    prior_strength: float = 10.0,
    model_a: str = "user_item_blend",
    model_b: str = "item_smoothed",
    n_bootstrap: int = 5000,
    seed: int = 271828,
) -> YahooRandomizedPairBootstrap:
    """Paired user-cluster bootstrap for randomized Brier difference.

    Difference is Brier(A) - Brier(B), so negative values favor model A.
    Models are fitted once on the logged-fit split for ``fit_seed``.
    """
    train = np.asarray(train, dtype=int)
    randomized = np.asarray(randomized, dtype=int)

    if train.ndim != 2 or train.shape[1] != 3:
        raise ValueError("train must have shape (n, 3)")
    if randomized.ndim != 2 or randomized.shape[1] != 3:
        raise ValueError("randomized must have shape (n, 3)")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")

    cohort_train = restrict_logged_to_randomized_users(train, randomized)
    split = split_logged_triplets_by_user(
        cohort_train,
        holdout_fraction=holdout_fraction,
        seed=fit_seed,
    )
    fitted = fit_yahoo_probability_models(
        split.fit_triplets,
        positive_threshold=positive_threshold,
        prior_strength=prior_strength,
    )
    predictions = predict_yahoo_probability_models(fitted, randomized)

    if model_a not in predictions or model_b not in predictions:
        raise ValueError("requested model is not present in candidate set")

    y = (randomized[:, 2] >= positive_threshold).astype(float)
    loss_diff = (
        np.square(predictions[model_a] - y)
        - np.square(predictions[model_b] - y)
    )

    users = np.unique(randomized[:, 0])
    user_losses: list[np.ndarray] = []
    for user in users:
        user_losses.append(loss_diff[randomized[:, 0] == user])

    observed_difference = float(np.mean(loss_diff))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap, dtype=float)

    for index in range(n_bootstrap):
        sampled = rng.integers(0, len(user_losses), size=len(user_losses))
        boot[index] = float(
            np.mean(
                np.concatenate([user_losses[i] for i in sampled])
            )
        )

    lower, upper = np.quantile(boot, [0.025, 0.975])

    return YahooRandomizedPairBootstrap(
        model_a=model_a,
        model_b=model_b,
        observed_difference=observed_difference,
        lower_95=float(lower),
        upper_95=float(upper),
        probability_a_better=float(np.mean(boot < 0.0)),
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
