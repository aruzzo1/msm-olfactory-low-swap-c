"""Rolling Augmented Dickey-Fuller (ADF) and KPSS tests on a transformed chromatogram, and per-sample summaries.

The two tests have opposite null hypotheses, so together they map local stability along the run:

* ADF  H0: unit root (non-stationary).  A low p-value is evidence the window IS stationary.
* KPSS H0: level stationarity.          A low p-value is evidence the window is NOT stationary.

statsmodels interpolates KPSS p-values from a lookup table and clips them to [0.01, 0.10]. As a result,
``frac_kpss_p_lt_0_01`` (strictly below 0.01) is always 0 on real data.
"""

from __future__ import annotations

import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss

from msm_olfactory.config import P_THRESHOLD, P_THRESHOLD_STRICT
from msm_olfactory.utils import quiet_numerics


def _p_value_or_nan(test, segment: np.ndarray, **kwargs) -> float:
    """p-value of one test on one window; NaN if the test fails."""
    try:
        return float(test(segment, **kwargs)[1])
    except Exception:
        return np.nan


def rolling_stationarity_tests(y: np.ndarray, time: np.ndarray, window_size: int, step: int) -> dict[str, np.ndarray]:
    """Run ADF and KPSS on sliding windows ``y[start:start + window_size]`` for ``start = 0, step, 2*step, ...``.

    Parameters
    ----------
    y
        Transformed series (log1p + z-score), in time order.
    time
        Retention times aligned with ``y``.
    window_size, step
        Window length and stride in points.

    Returns
    -------
    dict
        ``times`` (mean retention time of each window), ``adf_p`` and ``kpss_p`` arrays of equal length. A window whose
        test fails gets NaN.

    Raises
    ------
    ValueError
        If ``y`` and ``time`` differ in length.
    """
    y, time = np.asarray(y, dtype=float), np.asarray(time, dtype=float)
    if len(y) != len(time):
        raise ValueError("y and time must have the same length")
    centres, adf_p, kpss_p = [], [], []
    with quiet_numerics():
        for start in range(0, len(y) - window_size + 1, step):
            segment = y[start : start + window_size]
            centres.append(time[start : start + window_size].mean())
            adf_p.append(_p_value_or_nan(adfuller, segment, autolag="AIC"))
            kpss_p.append(_p_value_or_nan(kpss, segment, regression="c", nlags="auto"))
    return {"times": np.array(centres), "adf_p": np.array(adf_p), "kpss_p": np.array(kpss_p)}


def compute_stationarity_features(sample_id: str, adf_p: np.ndarray, kpss_p: np.ndarray) -> dict | None:
    """Summarise rolling ADF/KPSS p-values into per-sample features (source Step 11).

    Only windows where both tests produced a p-value are used.

    Returns
    -------
    dict or None
        ``n_station_windows``; the fractions of windows with ADF or KPSS p below 0.05 and 0.01;
        ``frac_both_p_lt_0_05`` (both nulls rejected, meaning the tests disagree about the window);
        ``frac_adf_high_kpss_low`` (ADF cannot reject a unit root AND KPSS rejects stationarity, i.e. strong evidence of
        non-stationarity); and the mean and median of each p-value. ``None`` if no window has both p-values.
    """
    adf_p, kpss_p = np.asarray(adf_p, dtype=float), np.asarray(kpss_p, dtype=float)
    valid = np.isfinite(adf_p) & np.isfinite(kpss_p)
    if not valid.any():
        return None
    a, k = adf_p[valid], kpss_p[valid]
    return {
        "sample_id": sample_id,
        "n_station_windows": int(valid.sum()),
        "frac_adf_p_lt_0_05": float(np.mean(a < P_THRESHOLD)),
        "frac_adf_p_lt_0_01": float(np.mean(a < P_THRESHOLD_STRICT)),
        "frac_kpss_p_lt_0_05": float(np.mean(k < P_THRESHOLD)),
        "frac_kpss_p_lt_0_01": float(np.mean(k < P_THRESHOLD_STRICT)),
        "frac_both_p_lt_0_05": float(np.mean((a < P_THRESHOLD) & (k < P_THRESHOLD))),
        "frac_adf_high_kpss_low": float(np.mean((a > P_THRESHOLD) & (k < P_THRESHOLD))),
        "mean_adf_p": float(np.mean(a)),
        "median_adf_p": float(np.median(a)),
        "mean_kpss_p": float(np.mean(k)),
        "median_kpss_p": float(np.median(k)),
    }
