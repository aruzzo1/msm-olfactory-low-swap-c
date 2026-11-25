"""Markov switching model: transform, fit, output shapes, orientation and reproducibility."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from msm_olfactory import msm
from msm_olfactory.preprocessing import log_zscore_transform


def test_log_zscore_transform():
    y = log_zscore_transform(np.array([-5.0, 0.0, 10.0, 100.0, 1000.0]))
    assert y.mean() == pytest.approx(0.0, abs=1e-12)
    assert y.std() == pytest.approx(1.0)
    assert y[0] == y[1]  # negative intensity clipped to 0 before log1p


def test_transform_of_constant_series_is_finite():
    # The source's `std > 0` guard is defeated by rounding (std ~ 1e-16), so a constant series maps to a constant
    # rather than 0. Kept as in the source; it cannot occur for real chromatograms. The output stays finite.
    out = log_zscore_transform(np.full(10, 7.0))
    assert np.all(np.isfinite(out)) and np.ptp(out) == 0


def test_smoothed_probabilities_accepts_both_orientations():
    p = np.random.default_rng(0).random((50, 3))
    assert msm.smoothed_probabilities(SimpleNamespace(smoothed_marginal_probabilities=p), 3, 50).shape == (50, 3)
    assert msm.smoothed_probabilities(SimpleNamespace(smoothed_marginal_probabilities=p.T), 3, 50).shape == (50, 3)
    with pytest.raises(RuntimeError):
        msm.smoothed_probabilities(SimpleNamespace(smoothed_marginal_probabilities=p), 3, 49)


def test_msm_recovers_two_regimes(fast_settings, two_regime_series):
    y, states = two_regime_series
    fit = msm.fit_markov_switching_model("S1", y, replace(fast_settings, k_regimes=2), seed=1)
    assert fit.k_used == 2 and fit.regime_probs.shape == (len(y), 2) and fit.regimes.shape == (len(y),)
    # regimes are only identified up to relabelling: accept either assignment
    agreement = max(np.mean(fit.regimes == states), np.mean(fit.regimes == 1 - states))
    assert agreement > 0.95
    # statsmodels convention: P[i, j] = P(S_t = i | S_{t-1} = j), so columns sum to 1
    np.testing.assert_allclose(fit.transition.sum(axis=0), 1.0, atol=1e-6)
    assert np.all(np.diag(fit.transition) > 0.9)  # persistent regimes


def test_seeded_fit_is_reproducible(fast_settings, two_regime_series):
    y, _ = two_regime_series
    a = msm.fit_markov_switching_model("S1", y, fast_settings, seed=11)
    b = msm.fit_markov_switching_model("S1", y, fast_settings, seed=11)
    assert a.llf == b.llf
    np.testing.assert_array_equal(a.regimes, b.regimes)


def test_features_pad_two_regime_fit_to_three():
    fit = msm.MsmFit(
        "S", 4, 2, -1.0, 2.0, 3.0, np.zeros((4, 2)), np.array([0, 0, 1, 1]), np.array([[0.9, 0.2], [0.1, 0.8]])
    )
    row = msm.extract_msm_features(fit)
    assert row["regime_frac_0"] == 0.5 and np.isnan(row["regime_frac_2"])
    assert row["trans_1_0"] == 0.1 and np.isnan(row["trans_2_2"]) and np.isnan(row["trans_0_2"])
    assert len([k for k in row if k.startswith("trans_")]) == 9
    assert len(row) == 18  # sample_id + 17 MSM features, as in the source


def test_too_short_series_raises(fast_settings):
    with pytest.raises(ValueError, match="at least"):
        msm.fit_markov_switching_model("S", np.ones(10), fast_settings, seed=0)
