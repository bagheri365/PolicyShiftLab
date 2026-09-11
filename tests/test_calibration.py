import numpy as np
import pytest

from policyshiftlab.calibration import (
    bootstrap_calibration_bands,
    calibration_bins,
    expected_calibration_error,
    quantile_bin_edges,
)


def test_perfectly_calibrated_two_bin_example_has_zero_ece() -> None:
    y = np.array([0, 0, 1, 1], dtype=float)
    p = np.array([0.0, 0.0, 1.0, 1.0], dtype=float)

    ece = expected_calibration_error(
        y,
        p,
        n_bins=2,
        strategy="uniform",
    )

    assert ece == pytest.approx(0.0)


def test_weighted_calibration_bins_match_manual_calculation() -> None:
    y = np.array([0, 1, 0, 1], dtype=float)
    p = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)
    w = np.array([1.0, 3.0, 2.0, 2.0])

    bins = calibration_bins(
        y,
        p,
        sample_weight=w,
        n_bins=2,
        strategy="uniform",
    )

    assert bins.mean_predicted[0] == pytest.approx(
        (1.0 * 0.1 + 3.0 * 0.2) / 4.0
    )
    assert bins.fraction_positive[0] == pytest.approx(3.0 / 4.0)
    assert bins.bin_weight[0] == pytest.approx(4.0)
    assert bins.bin_count[0] == 2


def test_quantile_bins_have_expected_total_count() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=101)
    p = rng.random(101)

    bins = calibration_bins(
        y,
        p,
        n_bins=10,
        strategy="quantile",
    )

    assert int(np.sum(bins.bin_count)) == 101


def test_weighted_ece_changes_when_weights_change() -> None:
    # The low-score bin is intentionally much more miscalibrated than the
    # high-score bin, so reweighting that bin must change aggregate ECE.
    y = np.array([0, 1, 1, 1], dtype=float)
    p = np.array([0.1, 0.1, 0.9, 0.9], dtype=float)

    unweighted = expected_calibration_error(
        y,
        p,
        n_bins=2,
        strategy="uniform",
    )
    weighted = expected_calibration_error(
        y,
        p,
        sample_weight=np.array([10.0, 10.0, 1.0, 1.0]),
        n_bins=2,
        strategy="uniform",
    )

    assert weighted > unweighted


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_bins": 1},
        {"strategy": "bad"},
    ],
)
def test_invalid_configuration_is_rejected(kwargs) -> None:
    y = np.array([0, 1], dtype=float)
    p = np.array([0.2, 0.8], dtype=float)

    with pytest.raises(ValueError):
        calibration_bins(y, p, **kwargs)


def test_shared_bin_edges_are_reused_across_populations() -> None:
    target_p = np.array([0.05, 0.15, 0.35, 0.55, 0.75, 0.95])
    target_y = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    logged_mask = np.array([False, True, True, True, True, False])

    edges = quantile_bin_edges(target_p, n_bins=3)

    target_bins = calibration_bins(
        target_y,
        target_p,
        bin_edges=edges,
    )
    logged_bins = calibration_bins(
        target_y[logged_mask],
        target_p[logged_mask],
        bin_edges=edges,
    )

    assert target_bins.mean_predicted.size == 3
    assert logged_bins.mean_predicted.size <= 3
    assert int(np.sum(target_bins.bin_count)) == target_p.size
    assert int(np.sum(logged_bins.bin_count)) == int(np.sum(logged_mask))


def test_shared_edges_cover_zero_and_one() -> None:
    p = np.array([0.2, 0.4, 0.6, 0.8])

    edges = quantile_bin_edges(p, n_bins=2)

    assert edges[0] == pytest.approx(0.0)
    assert edges[-1] == pytest.approx(1.0)


def test_bootstrap_bands_have_expected_shape_and_order() -> None:
    rng = np.random.default_rng(4)
    p = np.linspace(0.01, 0.99, 400)
    y = rng.binomial(1, p)
    edges = quantile_bin_edges(p, n_bins=5)

    bands = bootstrap_calibration_bands(
        y,
        p,
        bin_edges=edges,
        n_bootstrap=100,
        seed=5,
    )

    assert bands.lower.shape == (5,)
    assert bands.upper.shape == (5,)
    assert np.all(bands.lower <= bands.upper)
    assert np.all(
        (bands.fraction_positive >= 0.0)
        & (bands.fraction_positive <= 1.0)
    )


def test_weighted_bootstrap_is_reproducible() -> None:
    rng = np.random.default_rng(8)
    p = np.linspace(0.01, 0.99, 300)
    y = rng.binomial(1, p)
    w = np.linspace(0.5, 3.0, 300)
    edges = quantile_bin_edges(p, n_bins=5)

    first = bootstrap_calibration_bands(
        y,
        p,
        sample_weight=w,
        bin_edges=edges,
        n_bootstrap=80,
        seed=10,
    )
    second = bootstrap_calibration_bands(
        y,
        p,
        sample_weight=w,
        bin_edges=edges,
        n_bootstrap=80,
        seed=10,
    )

    np.testing.assert_allclose(first.lower, second.lower)
    np.testing.assert_allclose(first.upper, second.upper)


@pytest.mark.parametrize(
    ("n_bootstrap", "confidence_level"),
    [
        (1, 0.95),
        (10, 0.0),
        (10, 1.0),
    ],
)
def test_invalid_bootstrap_configuration_is_rejected(
    n_bootstrap,
    confidence_level,
) -> None:
    y = np.array([0, 0, 1, 1], dtype=float)
    p = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)
    edges = np.array([0.0, 0.5, 1.0])

    with pytest.raises(ValueError):
        bootstrap_calibration_bands(
            y,
            p,
            bin_edges=edges,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
        )
