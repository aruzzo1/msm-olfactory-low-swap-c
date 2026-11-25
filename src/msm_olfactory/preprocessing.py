"""Signal preprocessing shared by the MSM and the stationarity tests."""

from __future__ import annotations

import numpy as np


def log_zscore_transform(intensity: np.ndarray) -> np.ndarray:
    """Compress and standardise a raw TIC intensity series: ``zscore(log1p(max(x, 0)))``.

    Parameters
    ----------
    intensity
        Raw total-ion intensities (counts) in retention-time order.

    Returns
    -------
    numpy.ndarray
        Transformed series with mean 0 and standard deviation 1.

    Notes
    -----
    The log compression stops the few very large peaks (up to about 10^8 counts) from dominating the regime model.
    The z-score puts every sample on a common scale before fitting. The ``std > 0`` guard is kept from the source.
    Because of floating-point rounding, a constant input has a standard deviation of about 1e-16 rather than 0, so it
    maps to a constant (not necessarily 0). Real chromatograms are never constant.
    """
    y = np.log1p(np.clip(np.asarray(intensity, dtype=float), a_min=0, a_max=None))
    std = y.std()
    return (y - y.mean()) / (std if std > 0 else 1.0)
