"""Computational trade study (SWaP-C): classifier cost and the feature-extraction cost each classifier depends on.

The source's Step 19 timed only classifier forward passes. Here the same models are timed with a more stable protocol
(median of many repetitions, one CPU thread as a proxy for a CPU-only edge device), and the per-sample cost of
producing their inputs is also measured: MSM fit, rolling ADF/KPSS and TIC-image assembly. Forward-pass time does not
depend on trained weights, so freshly initialised models are timed on random inputs of the correct shape.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from msm_olfactory import data, images, msm, stationarity
from msm_olfactory.config import CNN_NAME, MLP_NAME, MULTIMODAL_NAME, N_IMAGE_ROWS, Settings
from msm_olfactory.preprocessing import log_zscore_transform
from msm_olfactory.utils import quiet_numerics

BYTES_PER_FLOAT32 = 4
TIMING_WARMUP = 10
#: Intensity values per raw TIC row: retention time and intensity.
RAW_VALUES_PER_POINT = 2


def _median_seconds(fn: Callable[[], object], reps: int, warmup: int = TIMING_WARMUP) -> float:
    """Median wall-clock seconds of ``fn()`` over ``reps`` calls, after ``warmup`` untimed calls."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def benchmark_classifier_inference(
    tab_dim: int,
    image_width: int,
    batch_sizes: Sequence[int] = (1, 8),
    reps: int = 200,
    threads: int = 1,
    seed: int = 0,
) -> list[dict]:
    """Median forward-pass seconds per batch for the three models on ``threads`` CPU threads.

    The multimodal ResNet18 is about 100x slower, so it is timed with a quarter of the repetitions.

    Returns
    -------
    list of dict
        One row per (model, batch size): ``model, batch_size, threads, median_seconds_per_batch, parameters``.
    """
    import torch

    from msm_olfactory.deep_models import build_models, count_parameters

    torch.manual_seed(seed)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        rows = []
        for name, model in build_models(tab_dim, image_width).items():
            model.eval()
            n_reps = reps // 4 if name == MULTIMODAL_NAME else reps
            for batch in batch_sizes:
                img, tab = torch.rand(batch, 1, N_IMAGE_ROWS, image_width), torch.rand(batch, tab_dim)
                with torch.no_grad():
                    seconds = _median_seconds(lambda m=model, i=img, t=tab: m(i, t), reps=n_reps)
                rows.append(
                    {
                        "model": name,
                        "batch_size": batch,
                        "threads": threads,
                        "median_seconds_per_batch": seconds,
                        "parameters": count_parameters(model),
                    }
                )
        return rows
    finally:
        torch.set_num_threads(previous_threads)


def benchmark_feature_extraction(tic: pd.DataFrame, sample_ids: Sequence[str], settings: Settings) -> pd.DataFrame:
    """Single-process wall-clock seconds per sample for each preprocessing stage.

    Returns
    -------
    pandas.DataFrame
        Per sample: ``msm_fit_seconds``, ``rolling_adf_kpss_seconds``, ``image_assembly_seconds`` and
        ``intensity_resample_seconds`` (the only preprocessing an intensity-only image would need).
    """
    rows = []
    for sid in sample_ids:
        time_axis, intensity = data.get_sample_series(tic, sid)
        with quiet_numerics():
            t0 = time.perf_counter()
            y = log_zscore_transform(intensity)
            fit = msm.fit_markov_switching_model(sid, intensity, settings, seed=settings.base_seed)
            t1 = time.perf_counter()
            rolling = stationarity.rolling_stationarity_tests(
                y, time_axis, settings.station_window, settings.station_step
            )
            t2 = time.perf_counter()
        images.build_tic_image(
            time_axis,
            intensity,
            fit.regimes,
            fit.k_used,
            rolling["times"],
            rolling["adf_p"],
            rolling["kpss_p"],
            settings.image_width,
        )
        t3 = time.perf_counter()
        grid = np.linspace(time_axis[0], time_axis[-1], settings.image_width)
        resample = _median_seconds(
            lambda g=grid, t=time_axis, v=intensity: images.normalize_row_to_01(np.interp(g, t, v)), reps=50, warmup=5
        )
        rows.append(
            {
                "sample_id": sid,
                "n_points": len(intensity),
                "msm_fit_seconds": t1 - t0,
                "rolling_adf_kpss_seconds": t2 - t1,
                "image_assembly_seconds": t3 - t2,
                "intensity_resample_seconds": resample,
            }
        )
    return pd.DataFrame(rows)


def summarize_compute_tradeoffs(
    classifier_rows: list[dict], extraction: pd.DataFrame, tab_dim: int, image_width: int, raw_points: int
) -> dict:
    """Combine the timings into the trade-study summary, with each value labelled measured or calculated."""

    def latency(model: str, batch: int) -> float:
        return next(
            r["median_seconds_per_batch"] for r in classifier_rows if r["model"] == model and r["batch_size"] == batch
        )

    params = {r["model"]: r["parameters"] for r in classifier_rows}
    models = (CNN_NAME, MLP_NAME, MULTIMODAL_NAME)
    msm_s = float(extraction["msm_fit_seconds"].median())
    rolling_s = float(extraction["rolling_adf_kpss_seconds"].median())
    image_s = float(extraction["image_assembly_seconds"].median())
    cnn_inputs = N_IMAGE_ROWS * image_width
    return {
        "measured": {
            "forward_seconds_per_batch8_1thread": {m: latency(m, 8) for m in models},
            "forward_seconds_per_sample_1thread": {m: latency(m, 1) for m in models},
            "feature_extraction_seconds_per_sample_median": {
                "msm_fit": msm_s,
                "rolling_adf_kpss": rolling_s,
                "image_assembly": image_s,
                "intensity_only_resample": float(extraction["intensity_resample_seconds"].median()),
            },
            "n_samples_timed": len(extraction),
        },
        "calculated": {
            "speedup_cnn_over_mlp_batch8": latency(CNN_NAME, 8) / latency(MLP_NAME, 8),
            "parameters": params,
            "parameter_reduction_cnn_over_mlp": params[CNN_NAME] / params[MLP_NAME],
            "weights_bytes_float32": {m: p * BYTES_PER_FLOAT32 for m, p in params.items()},
            "classifier_inputs_per_sample": {CNN_NAME: cnn_inputs, MLP_NAME: tab_dim},
            "classifier_input_reduction": cnn_inputs / tab_dim,
            "classifier_input_bytes_float32": {
                CNN_NAME: cnn_inputs * BYTES_PER_FLOAT32,
                MLP_NAME: tab_dim * BYTES_PER_FLOAT32,
            },
            "raw_tic_values_per_sample": RAW_VALUES_PER_POINT * raw_points,
            "raw_tic_bytes_float32": RAW_VALUES_PER_POINT * raw_points * BYTES_PER_FLOAT32,
            "end_to_end_seconds_per_sample": {
                "MLP path (MSM + ADF/KPSS + MLP)": msm_s + rolling_s + latency(MLP_NAME, 1),
                "CNN path as implemented (MSM + ADF/KPSS + image + CNN)": msm_s
                + rolling_s
                + image_s
                + latency(CNN_NAME, 1),
            },
        },
        "notes": [
            "Timings are machine-specific (desktop CPU, 1 PyTorch thread); ratios are more portable than absolute "
            "values.",
            "Both implemented classifiers need the MSM fit and rolling ADF/KPSS: the MLP's 30 inputs come from "
            "them, and rows 1-3 of the CNN's image are MSM-regime and ADF/KPSS channels.",
            "The input-size reduction applies between feature extraction and the classifier. Either path must first "
            "acquire and process the full raw TIC on the device.",
        ],
    }
