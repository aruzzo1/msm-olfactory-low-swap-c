"""End-to-end behaviour on SYNTHETIC data: the software runs, is reproducible, caches and fails clearly.

These tests check the software, not any scientific result (the synthetic labels are random).
"""

import importlib.util
import json
import logging
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from msm_olfactory import data
from msm_olfactory import visualization as viz
from msm_olfactory.config import MODEL_NAMES, PROJECT_ROOT
from msm_olfactory.pipeline import run_pipeline
from msm_olfactory.sample_dynamics import analyse_sample, compute_sample_dynamics, sample_seed


def test_sample_seed_is_stable_and_distinct():
    assert sample_seed(42, "S001") == sample_seed(42, "S001")
    assert sample_seed(42, "S001") != sample_seed(42, "S002")


def test_parallel_and_serial_dynamics_are_identical(synthetic_root, tmp_path, fast_settings):
    tic = data.build_tic_table(synthetic_root)
    serial = compute_sample_dynamics(tic, fast_settings, tmp_path / "serial.pkl")
    parallel = compute_sample_dynamics(tic, replace(fast_settings, workers=2), tmp_path / "parallel.pkl")
    for sid, d in serial.items():
        assert d.msm_features == parallel[sid].msm_features
        assert d.stationarity_features == parallel[sid].stationarity_features


def test_failed_fits_are_recorded_and_all_failures_raise(synthetic_root, tmp_path, fast_settings):
    tic = data.build_tic_table(synthetic_root)
    impossible = replace(fast_settings, msm_min_points=10**6)
    time, intensity = data.get_sample_series(tic, tic["sample_id"].iloc[0])
    assert analyse_sample("X", time, intensity, impossible).failure is not None
    with pytest.raises(RuntimeError, match="any sample"):
        compute_sample_dynamics(tic, impossible, tmp_path / "x.pkl")


def test_pipeline_end_to_end_on_synthetic_data(synthetic_root, tmp_path, fast_settings, caplog):
    summary = run_pipeline(synthetic_root, tmp_path / "out", fast_settings, is_synthetic=True)
    assert summary["data_root_is_synthetic"] is True
    assert summary["ingestion"]["samples"] == 12 and summary["msm"]["failed_samples"] == []
    assert set(summary["deep_models"]["best_val_f1"]) == set(MODEL_NAMES)
    # Only synthetic illustrations and aggregate figures are public; the sample-level panel goes to the cache.
    public_figures = sorted(p.name for p in (tmp_path / "out" / "figures").glob("*.png"))
    assert len(public_figures) == 9 and not any(name.startswith("05_") for name in public_figures)
    assert (
        tmp_path / "out" / "cache" / "sample_level_figures" / "05_tic_image_panel_cancer_vs_noncancer.png"
    ).is_file()
    summary_text = (tmp_path / "out" / "results" / "run_summary.json").read_text()
    doc = json.loads(summary_text)
    assert doc["pca"]["combat_output_nan_values"] == 0
    # Published results are de-identified: no sample ID appears anywhere in the run summary.
    sample_ids = data.build_tic_table(synthetic_root)["sample_id"].unique()
    assert not [sid for sid in sample_ids if sid in summary_text]
    assert doc["image_panel_samples"]["n_cancer"] == doc["image_panel_samples"]["n_non_cancer"] == 4

    # A second run reuses the cached per-sample fits and reproduces every analytical result exactly.
    with caplog.at_level(logging.INFO, logger="msm_olfactory"):
        again = run_pipeline(synthetic_root, tmp_path / "out", fast_settings, is_synthetic=True)
    assert any("Loaded cached MSM" in r.getMessage() for r in caplog.records)
    for key in ("ingestion", "msm", "tabular_baselines", "pca", "image_panel_samples"):
        assert json.dumps(again[key], default=str) == json.dumps(summary[key], default=str)
    for key in ("best_val_f1", "best_epoch", "final_epoch_val_f1"):
        assert again["deep_models"][key] == summary["deep_models"][key]


def test_cli_reports_missing_data_clearly(tmp_path):
    spec = importlib.util.spec_from_file_location("run_analysis", PROJECT_ROOT / "scripts" / "run_analysis.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    assert cli.parse_args([]).seed == 42
    assert cli.main(["--data", str(tmp_path / "missing"), "--output", str(tmp_path / "out")]) == 1


def test_package_import_does_not_load_torch():
    # The parallel MSM workers import the package; PyTorch must stay a lazy dependency.
    code = "import sys, msm_olfactory; print('torch' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"


def test_plot_functions_render(tmp_path: Path):
    t = np.linspace(1, 40, 300)
    y = np.abs(np.sin(t)) * 1e6
    viz.save_figure(viz.plot_msm_regimes(t, y, (t > 20).astype(int), 2, "X"), tmp_path / "a.png")
    importance = pd.DataFrame(
        {"feature": ["a", "mean_adf_p"], "importance": [0.6, 0.4], "group": ["MSM", "Stationarity"]}
    )
    viz.save_figure(viz.plot_feature_importance(importance), tmp_path / "b.png")
    history = {"m": [{"epoch": 1, "val_f1": 0.5, "val_loss": 0.7}, {"epoch": 2, "val_f1": 0.6, "val_loss": 0.6}]}
    viz.save_figure(viz.plot_training_curves(history, n_val=20, reference_f1=0.75), tmp_path / "c.png")
    assert all((tmp_path / f).stat().st_size > 1000 for f in ("a.png", "b.png", "c.png"))
