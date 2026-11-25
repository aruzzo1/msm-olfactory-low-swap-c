"""Rolling ADF/KPSS tests, their per-sample summaries, and the TIC image."""

import numpy as np
import pytest

from msm_olfactory import images, stationarity


def test_rolling_window_count_and_centres():
    t = np.arange(1000, dtype=float)
    out = stationarity.rolling_stationarity_tests(np.random.default_rng(0).normal(size=1000), t, 200, 20)
    assert len(out["times"]) == len(out["adf_p"]) == len(out["kpss_p"]) == (1000 - 200) // 20 + 1
    assert out["times"][0] == pytest.approx(t[:200].mean())


def test_rolling_tests_reject_mismatched_inputs():
    with pytest.raises(ValueError, match="same length"):
        stationarity.rolling_stationarity_tests(np.zeros(10), np.zeros(9), 5, 1)


def test_tests_point_in_expected_directions():
    rng = np.random.default_rng(1)
    t = np.arange(600.0)
    white = stationarity.rolling_stationarity_tests(rng.normal(size=600), t, 200, 100)
    walk = stationarity.rolling_stationarity_tests(np.cumsum(rng.normal(size=600)), t, 200, 100)
    assert np.all(white["adf_p"] < 0.05)  # stationary noise: ADF rejects the unit root
    assert np.median(walk["kpss_p"]) < 0.05  # random walk: KPSS rejects stationarity
    assert np.all((white["kpss_p"] >= 0.01) & (white["kpss_p"] <= 0.1))  # statsmodels clips KPSS p-values


def test_stationarity_features_from_hand_computed_windows():
    adf = np.array([0.001, 0.03, 0.2, np.nan])
    kpss = np.array([0.02, 0.1, 0.01, 0.05])
    f = stationarity.compute_stationarity_features("S", adf, kpss)
    assert f["n_station_windows"] == 3  # the NaN window is excluded
    assert f["frac_adf_p_lt_0_05"] == pytest.approx(2 / 3)
    assert f["frac_adf_p_lt_0_01"] == pytest.approx(1 / 3)
    assert f["frac_both_p_lt_0_05"] == pytest.approx(1 / 3)
    assert f["frac_adf_high_kpss_low"] == pytest.approx(1 / 3)
    assert len(f) == 12  # sample_id + 11 stationarity features, as in the source
    assert stationarity.compute_stationarity_features("S", np.array([np.nan]), np.array([0.1])) is None


def test_normalize_row_handles_nan_and_constant():
    assert np.all(images.normalize_row_to_01(np.full(5, np.nan)) == 0)
    assert np.all(images.normalize_row_to_01(np.full(5, 3.0)) == 0)
    r = images.normalize_row_to_01(np.array([0.0, 1.0, np.nan, 2.0, 100.0]))
    assert np.nanmin(r) == 0.0 and np.nanmax(r) == 1.0


def test_build_tic_image_shape_and_roundtrip(tmp_path):
    t = np.linspace(1, 40, 800)
    y = np.abs(np.sin(t)) * 1e6
    regimes = (t > 20).astype(int) * 2
    wt = np.linspace(3, 38, 30)
    img, grid = images.build_tic_image(t, y, regimes, 3, wt, np.linspace(0, 1, 30), np.full(30, 0.05), 256)
    assert img.shape == (4, 256) and img.dtype == np.uint8 and grid.shape == (256,)
    assert set(np.unique(img[1])) <= {0, 255}  # regimes 0 and 2 only -> scaled to 0 and 1
    back = images.load_tic_image(images.save_tic_image(img, tmp_path / "x.png"))
    np.testing.assert_allclose(back * 255, img, atol=0.5)
