"""Computational trade study: timing helpers, derived ratios, provenance table and figures."""

import numpy as np
import pandas as pd
import pytest

from msm_olfactory import benchmarking, data
from msm_olfactory import visualization as viz
from msm_olfactory.config import MODEL_NAMES
from msm_olfactory.reported import REPORTED_THREE_MODEL_RUNS, THREE_MODEL_NAMES


def test_classifier_benchmark_covers_models_and_batches():
    rows = benchmarking.benchmark_classifier_inference(30, 256, batch_sizes=(1, 2), reps=4)
    assert len(rows) == 6
    params = {r["model"]: r["parameters"] for r in rows}
    assert params["TIC-image CNN"] == 1_072_066 and params["MSM-feature MLP"] == 4_130
    assert all(r["median_seconds_per_batch"] > 0 for r in rows)


def test_feature_extraction_benchmark_and_summary(synthetic_root, fast_settings):
    tic = data.build_tic_table(synthetic_root)
    extraction = benchmarking.benchmark_feature_extraction(tic, sorted(tic["sample_id"].unique())[:1], fast_settings)
    assert {"msm_fit_seconds", "rolling_adf_kpss_seconds", "image_assembly_seconds"} <= set(extraction.columns)
    rows = benchmarking.benchmark_classifier_inference(30, 256, batch_sizes=(1, 8), reps=2)
    summary = benchmarking.summarize_compute_tradeoffs(rows, extraction, tab_dim=30, image_width=256, raw_points=500)
    calculated = summary["calculated"]
    assert calculated["classifier_input_reduction"] == pytest.approx(1024 / 30)
    assert calculated["parameter_reduction_cnn_over_mlp"] == pytest.approx(1_072_066 / 4_130)
    assert calculated["raw_tic_values_per_sample"] == 1000
    msm_seconds = summary["measured"]["feature_extraction_seconds_per_sample_median"]["msm_fit"]
    assert min(calculated["end_to_end_seconds_per_sample"].values()) >= msm_seconds


def test_reported_values_are_well_formed():
    assert THREE_MODEL_NAMES == MODEL_NAMES
    assert len(REPORTED_THREE_MODEL_RUNS) == 6
    assert all(len(r["f1"]) == 3 and all(0 <= v <= 1 for v in r["f1"]) for r in REPORTED_THREE_MODEL_RUNS)


def test_seed_study_and_tradeoff_figures_render(tmp_path):
    rng = np.random.default_rng(0)
    table = pd.DataFrame(
        {
            "always_cancer_val_f1": [0.75] * 4,
            **{f"{m} | {k}": rng.random(4) for m in MODEL_NAMES for k in ("best_val_f1", "final_val_f1")},
        }
    )
    viz.save_figure(viz.plot_seed_study(table), tmp_path / "seed.png")
    summary = {
        "measured": {
            "forward_seconds_per_batch8_1thread": dict(zip(MODEL_NAMES, [3e-3, 4e-5, 0.3], strict=True)),
            "forward_seconds_per_sample_1thread": dict(zip(MODEL_NAMES, [6e-4, 2e-5, 0.04], strict=True)),
            "feature_extraction_seconds_per_sample_median": {
                "msm_fit": 12.0,
                "rolling_adf_kpss": 1.5,
                "image_assembly": 5e-4,
                "intensity_only_resample": 1e-4,
            },
        },
        "calculated": {
            "parameters": dict(zip(MODEL_NAMES, [1_072_066, 4_130, 11_396_931], strict=True)),
            "raw_tic_values_per_sample": 14_670,
            "classifier_inputs_per_sample": {MODEL_NAMES[0]: 1024, MODEL_NAMES[1]: 30},
        },
    }
    viz.save_figure(viz.plot_swap_c(summary), tmp_path / "swap.png")
    assert (tmp_path / "seed.png").stat().st_size > 5000 and (tmp_path / "swap.png").stat().st_size > 5000
