"""Per-sample dynamic analysis: one MSM fit and one set of rolling stationarity tests per chromatogram.

This is the computationally expensive stage (about 14 s of CPU per sample). It runs in parallel processes, seeds each
sample independently of processing order, and caches the results on disk. The same fit is reused for the feature
table and for the TIC image (the source refitted the MSM when building images).
"""

from __future__ import annotations

import json
import logging
import pickle
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from msm_olfactory import data, images, msm, stationarity
from msm_olfactory.config import Settings
from msm_olfactory.preprocessing import log_zscore_transform
from msm_olfactory.utils import quiet_numerics

logger = logging.getLogger(__name__)

#: Log progress every this many completed samples.
PROGRESS_EVERY = 50


@dataclass
class SampleDynamics:
    """MSM and stationarity results for one chromatogram.

    ``failure`` holds the reason when the MSM could not be fitted; all other result fields are then ``None``.
    Stationarity fields are ``None`` when the series is shorter than ``settings.station_min_points``.
    """

    sample_id: str
    msm_features: dict | None = None
    regimes: np.ndarray | None = None
    k_used: int | None = None
    stationarity_features: dict | None = None
    window_times: np.ndarray | None = None
    adf_p: np.ndarray | None = None
    kpss_p: np.ndarray | None = None
    failure: str | None = None

    @property
    def ok(self) -> bool:
        """True if the MSM was fitted."""
        return self.failure is None


def sample_seed(base_seed: int, sample_id: str) -> int:
    """Deterministic per-sample seed, independent of processing order and number of workers."""
    return (base_seed + zlib.crc32(sample_id.encode("utf-8"))) % (2**31 - 1)


def analyse_sample(sample_id: str, time: np.ndarray, intensity: np.ndarray, settings: Settings) -> SampleDynamics:
    """Fit the MSM and run the rolling stationarity tests for one chromatogram.

    An MSM failure (series too short, or both the 3- and 2-regime fits failing) is recorded in
    ``SampleDynamics.failure`` rather than raised, as in the source, which skipped such samples.
    """
    with quiet_numerics():
        try:
            fit = msm.fit_markov_switching_model(
                sample_id, intensity, settings, seed=sample_seed(settings.base_seed, sample_id)
            )
        except ValueError as exc:
            return SampleDynamics(sample_id=sample_id, failure=str(exc))
        result = SampleDynamics(
            sample_id=sample_id,
            msm_features=msm.extract_msm_features(fit),
            regimes=fit.regimes.astype(np.int8),
            k_used=fit.k_used,
        )
        if len(intensity) >= settings.station_min_points:
            rolling = stationarity.rolling_stationarity_tests(
                log_zscore_transform(intensity), time, settings.station_window, settings.station_step
            )
            result.window_times = rolling["times"]
            result.adf_p, result.kpss_p = rolling["adf_p"], rolling["kpss_p"]
            result.stationarity_features = stationarity.compute_stationarity_features(
                sample_id, rolling["adf_p"], rolling["kpss_p"]
            )
    return result


def _analyse_job(job: tuple) -> SampleDynamics:
    """Unpack a ``(sample_id, time, intensity, settings)`` job for :class:`ProcessPoolExecutor`."""
    return analyse_sample(*job)


def _load_cache(cache_path: Path, cache_key: str, sample_ids: set[str]) -> dict[str, SampleDynamics] | None:
    if not cache_path.is_file():
        return None
    try:
        cached = pickle.loads(cache_path.read_bytes())
    except (pickle.UnpicklingError, EOFError, AttributeError, ModuleNotFoundError):
        logger.warning("Ignoring unreadable cache %s; recomputing.", cache_path.name)
        return None
    if cached.get("key") != cache_key or set(cached.get("samples", {})) != sample_ids:
        return None
    return cached["samples"]


def compute_sample_dynamics(tic: pd.DataFrame, settings: Settings, cache_path: Path) -> dict[str, SampleDynamics]:
    """Run :func:`analyse_sample` for every sample in ``tic``, in parallel, with an on-disk cache.

    Parameters
    ----------
    tic
        Long TIC table from :func:`msm_olfactory.data.build_tic_table`.
    settings
        MSM/stationarity settings and ``workers``. Only :meth:`Settings.msm_cache_key` fields affect the cache.
    cache_path
        Pickle file for the results (git-ignored output folder).

    Returns
    -------
    dict
        ``sample_id -> SampleDynamics``.

    Raises
    ------
    RuntimeError
        If the MSM failed for every sample, so the analysis cannot proceed.
    """
    sample_ids = list(tic["sample_id"].unique())
    cache_key = json.dumps(settings.msm_cache_key(), sort_keys=True)
    cached = _load_cache(cache_path, cache_key, set(sample_ids))
    if cached is not None:
        logger.info("Loaded cached MSM + stationarity results for %d samples (%s)", len(cached), cache_path.name)
        return cached

    jobs = [(sid, *data.get_sample_series(tic, sid), settings) for sid in sample_ids]
    logger.info(
        "Fitting %d-regime MSM + rolling ADF/KPSS for %d samples on %d worker(s)...",
        settings.k_regimes,
        len(jobs),
        settings.workers,
    )
    results: dict[str, SampleDynamics] = {}
    if settings.workers > 1:
        with ProcessPoolExecutor(max_workers=settings.workers) as pool:
            outputs = pool.map(_analyse_job, jobs, chunksize=1)
            for i, result in enumerate(outputs, start=1):
                results[result.sample_id] = result
                if i % PROGRESS_EVERY == 0:
                    logger.info("  %d/%d samples done", i, len(jobs))
    else:
        for job in jobs:
            result = _analyse_job(job)
            results[result.sample_id] = result

    if not any(r.ok for r in results.values()):
        raise RuntimeError("The MSM could not be fitted for any sample; check the input series.")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(pickle.dumps({"key": cache_key, "samples": results}))
    return results


def build_sample_tic_image(
    sample_id: str, tic: pd.DataFrame, dynamics: dict[str, SampleDynamics], settings: Settings
) -> np.ndarray:
    """TIC image for one sample from its cached dynamics.

    Raises
    ------
    ValueError
        If the sample has no MSM fit or no rolling-test series.
    """
    d = dynamics[sample_id]
    if not d.ok or d.window_times is None:
        raise ValueError(f"No MSM/stationarity series for {sample_id}")
    time, intensity = data.get_sample_series(tic, sample_id)
    image, _ = images.build_tic_image(
        time, intensity, d.regimes, d.k_used, d.window_times, d.adf_p, d.kpss_p, settings.image_width
    )
    return image
