"""Fixed-size 4 x W "TIC image" combining intensity, MSM regime and rolling stationarity (source Step 12).

Rows (row 0 at the bottom when plotted with ``origin="lower"``):

    0  raw intensity, linearly interpolated onto the grid
    1  MSM regime index, interpolated then rounded, divided by (K - 1)
    2  1 - ADF p-value  (bright = stationary window)
    3  1 - KPSS p-value (bright = non-stationary window)

Each row is scaled to [0, 1] between its own 1st and 99th percentiles and stored as uint8. Rows 1-3 are MSM and
stationarity channels, so a CNN trained on these images is NOT an intensity-only model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from msm_olfactory.config import IMAGE_ROW_PERCENTILES


def normalize_row_to_01(row: np.ndarray) -> np.ndarray:
    """Scale a row to [0, 1] between the 1st and 99th percentiles of its finite values (degenerate rows -> 0)."""
    row = np.asarray(row, dtype=float)
    finite = np.isfinite(row)
    if not finite.any():
        return np.zeros_like(row)
    lo, hi = np.percentile(row[finite], IMAGE_ROW_PERCENTILES)
    if hi <= lo:
        return np.zeros_like(row)
    return np.clip((row - lo) / (hi - lo), 0.0, 1.0)


def build_tic_image(
    time: np.ndarray,
    intensity: np.ndarray,
    regimes: np.ndarray,
    k_used: int,
    window_times: np.ndarray,
    adf_p: np.ndarray,
    kpss_p: np.ndarray,
    image_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the 4-row TIC image for one sample.

    Parameters
    ----------
    time, intensity
        Retention times and raw intensities.
    regimes, k_used
        Hard MSM regime labels per point and the number of regimes fitted.
    window_times, adf_p, kpss_p
        Rolling-test window centres and p-values.
    image_width
        Number of evenly spaced time bins.

    Returns
    -------
    tuple
        ``(image, grid)``: a uint8 array of shape ``(4, image_width)`` and the retention-time grid.
    """
    time = np.asarray(time, dtype=float)
    grid = np.linspace(time[0], time[-1], image_width)

    def interpolate(t_src, v_src):
        if len(t_src) < 2:
            return np.full_like(grid, np.nan, dtype=float)
        return np.interp(grid, t_src, v_src)

    intensity_row = interpolate(time, intensity)
    # Regimes are categorical: interpolate, then round back to the nearest regime index.
    if len(time) >= 2:
        regime_grid = np.rint(np.interp(grid, time, np.asarray(regimes, dtype=float)))
    else:
        regime_grid = np.zeros_like(grid)
    regime_row = regime_grid / (k_used - 1) if k_used > 1 else np.zeros_like(regime_grid)
    # Invert p-values so that strong test evidence (low p) is bright.
    adf_row = 1.0 - np.clip(interpolate(window_times, adf_p), 0.0, 1.0)
    kpss_row = 1.0 - np.clip(interpolate(window_times, kpss_p), 0.0, 1.0)

    rows = np.vstack([intensity_row, regime_row, adf_row, kpss_row])
    normalized = np.vstack([normalize_row_to_01(r) for r in rows])
    return (normalized * 255).astype(np.uint8), grid


def save_tic_image(image: np.ndarray, path: Path) -> Path:
    """Save a uint8 image as PNG, creating the folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
    return path


def load_tic_image(path: Path) -> np.ndarray:
    """Load a saved TIC image as float32 in [0, 1], shape ``(H, W)``."""
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
