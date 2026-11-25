"""End-to-end analysis pipeline reproducing the source notebook's Steps 4-19.

Stages (source step numbers in brackets)::

    load TICs + demographics, join labels                                   [4-7]   load_tic_table
    per-sample MSM fit + rolling ADF/KPSS (parallel, seeded, cached)         [8-11]  sample_dynamics
    feature tables                                                           [9-11a] features
    illustrative figures (SYNTHETIC chromatogram)                            [8, 10, 12b]
    tabular baselines + importances                                          [11b-c] tabular_models
    TIC-image panel                                                          [12c]
    PCA, ComBat by hospital, PCA before/after                                [13-15] batch_correction
    TIC-image CNN, MSM-feature MLP, gated multimodal ResNet18; Step-19 timing [16-19] deep_models

Per-sample tables, which contain clinical values, are written only to ``<output>/cache`` (git-ignored).
``<output>/results`` holds aggregate results, and ``<output>/figures`` the figures.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from msm_olfactory import batch_correction, data, features, tabular_models
from msm_olfactory import visualization as viz
from msm_olfactory.config import (
    CNN_NAME,
    DEFAULT_SETTINGS,
    IMAGE_PANEL_SAMPLES_PER_CLASS,
    MLP_NAME,
    MULTIMODAL_NAME,
    N_IMAGE_ROWS,
    PCA_RANDOM_STATE,
    Settings,
)
from msm_olfactory.evaluation import stratified_holdout_indices
from msm_olfactory.features import ModelMatrix
from msm_olfactory.images import build_tic_image, save_tic_image
from msm_olfactory.sample_dynamics import (
    SampleDynamics,
    analyse_sample,
    build_sample_tic_image,
    compute_sample_dynamics,
)
from msm_olfactory.synthetic import ILLUSTRATIVE_LABEL, make_illustrative_tic
from msm_olfactory.utils import write_json

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OutputPaths:
    """Output locations under one root: ``figures/``, ``results/`` (committed) and ``cache/`` (git-ignored)."""

    root: Path

    @property
    def figures(self) -> Path:
        """Generated figures."""
        return self.root / "figures"

    @property
    def results(self) -> Path:
        """Aggregate results (JSON/CSV)."""
        return self.root / "results"

    @property
    def cache(self) -> Path:
        """Per-sample intermediate data, fitted dynamics and TIC images."""
        return self.root / "cache"

    def cache_file(self, stem: str, settings: Settings, suffix: str) -> Path:
        """Cache file whose name records whether sample IDs were normalized (labels differ between the two modes)."""
        tag = "_normalized_ids" if settings.normalize_sample_ids else ""
        return self.cache / f"{stem}{tag}{suffix}"


# --- stage functions ----------------------------------------------------------------------------------------------


def load_tic_table(data_root: Path, paths: OutputPaths, settings: Settings) -> pd.DataFrame:
    """Load (or read from cache) the long TIC table joined with clinical metadata."""
    return data.load_or_build_tic_table(
        data_root, paths.cache_file("tic_all_with_metadata", settings, ".parquet"), settings.normalize_sample_ids
    )


def load_sample_dynamics(tic: pd.DataFrame, paths: OutputPaths, settings: Settings) -> dict[str, SampleDynamics]:
    """Per-sample MSM + stationarity results for the given settings (cached per MSM seed)."""
    stem = "dynamics" if settings.base_seed == DEFAULT_SETTINGS.base_seed else f"dynamics_seed{settings.base_seed}"
    return compute_sample_dynamics(tic, settings, paths.cache_file(stem, settings, ".pkl"))


def plot_illustrative_figures(paths: OutputPaths, settings: Settings) -> dict:
    """Figures 01, 02 and 04 (source Steps 8, 10 and 12b) drawn from a SYNTHETIC chromatogram.

    The figures only explain the method (MSM regimes, rolling ADF/KPSS, TIC image). A synthetic signal is used so that
    no source chromatogram is published. Every figure is titled "Synthetic illustrative example".
    """
    time, intensity = make_illustrative_tic()
    d = analyse_sample(ILLUSTRATIVE_LABEL, time, intensity, settings)
    if not d.ok:
        raise RuntimeError(f"MSM fit failed on the synthetic illustration: {d.failure}")
    viz.save_figure(
        viz.plot_msm_regimes(time, intensity, d.regimes, d.k_used, ILLUSTRATIVE_LABEL),
        paths.figures / "01_synthetic_illustration_msm_regimes.png",
    )
    viz.save_figure(
        viz.plot_msm_stationarity_panel(
            time,
            intensity,
            d.regimes,
            d.k_used,
            d.window_times,
            d.adf_p,
            d.kpss_p,
            ILLUSTRATIVE_LABEL,
            settings.station_window,
            settings.station_step,
        ),
        paths.figures / "02_synthetic_illustration_msm_adf_kpss.png",
    )
    image, _ = build_tic_image(
        time, intensity, d.regimes, d.k_used, d.window_times, d.adf_p, d.kpss_p, settings.image_width
    )
    viz.save_figure(
        viz.plot_tic_image(image, ILLUSTRATIVE_LABEL), paths.figures / "04_synthetic_illustration_tic_image.png"
    )
    return {"figures": ["01", "02", "04"], "data": "synthetic illustrative chromatogram (not source data)"}


def summarize_reference_sample(dynamics: dict[str, SampleDynamics], settings: Settings) -> dict:
    """Fit statistics of the source notebook's reference sample, used to check the reproduction against the original.

    Only model statistics are recorded (no ID, label, or signal values), so the summary does not identify the sample.
    """
    sample_id = settings.example_sample_id or sorted(dynamics)[0]
    d = dynamics[sample_id]
    features_row = d.msm_features
    return {
        "selection": "fixed reference sample used by the source notebook; identity not published",
        "n_points": features_row["n_points"],
        "k_used": features_row["k_used"],
        "llf": features_row["llf"],
        "aic": features_row["aic"],
        "bic": features_row["bic"],
        "regime_occupancy": {r: float(np.mean(d.regimes == r)) for r in range(d.k_used)},
        "n_windows": len(d.window_times),
        "adf_p_range": [float(np.nanmin(d.adf_p)), float(np.nanmax(d.adf_p))],
        "kpss_p_range": [float(np.nanmin(d.kpss_p)), float(np.nanmax(d.kpss_p))],
    }


def run_tabular_stage(features_full: pd.DataFrame, paths: OutputPaths, settings: Settings) -> dict:
    """Logistic regression and random forest baselines, importances and figure 03 (source Step 11b-c)."""
    result = tabular_models.run_tabular_baselines(features.prepare_tabular_features(features_full), settings)
    paths.results.mkdir(parents=True, exist_ok=True)
    result.importance.to_csv(paths.results / "random_forest_feature_importance.csv", index=False)
    result.coefficients.to_csv(paths.results / "logistic_regression_coefficients.csv", index=False)
    viz.save_figure(viz.plot_feature_importance(result.importance), paths.figures / "03_feature_importance_rf.png")
    m = result.metrics
    logger.info(
        "Tabular baselines, validation F1: LogReg (MSM group) %.3f | LogReg (all) %.3f | random forest %.3f",
        m.get("logreg_msm_group", {}).get("f1", float("nan")),
        m["logreg_all_features"]["f1"],
        m["random_forest_all_features"]["f1"],
    )
    return m


def plot_image_panel(
    tic: pd.DataFrame, dynamics: dict[str, SampleDynamics], paths: OutputPaths, settings: Settings
) -> dict:
    """Sample-level panel: TIC images of randomly chosen Cancer and Non-Cancer samples (source Step 12c).

    The figure shows individual real samples, so it is written to the git-ignored
    ``outputs/cache/sample_level_figures/`` folder for local inspection only and is never published. Panels are
    titled "Sample 1".."Sample n" (no sample IDs). Only the counts enter the run summary.
    """
    cancer_ids, noncancer_ids = data.pick_samples_by_label(tic, IMAGE_PANEL_SAMPLES_PER_CLASS, settings.random_state)
    logger.debug("Image-panel samples: cancer %s, non-cancer %s", cancer_ids, noncancer_ids)

    def anonymised(ids: list[str]) -> dict[str, np.ndarray]:
        return {f"Sample {k}": build_sample_tic_image(s, tic, dynamics, settings) for k, s in enumerate(ids, start=1)}

    viz.save_figure(
        viz.plot_tic_image_panel(anonymised(cancer_ids), anonymised(noncancer_ids)),
        paths.cache / "sample_level_figures" / "05_tic_image_panel_cancer_vs_noncancer.png",
    )
    return {
        "n_cancer": len(cancer_ids),
        "n_non_cancer": len(noncancer_ids),
        "selection": f"numpy default_rng({settings.random_state}) draw, as in the source; IDs not published",
    }


def run_batch_correction_stage(tic: pd.DataFrame, matrix: ModelMatrix, paths: OutputPaths) -> tuple[np.ndarray, dict]:
    """PCA of TIC summaries vs features, ComBat by hospital, PCA before/after; figures 06-08 (Steps 13-15).

    Returns
    -------
    tuple
        ``(X_combat, summary)``: the harmonized feature matrix and the PCA summary for the run report.
    """
    tic_summary = features.tic_summary_statistics(tic)
    tic_std = batch_correction.standardize_features(tic_summary[features.TIC_SUMMARY_COLUMNS].to_numpy(float))
    tic_pca, tic_evr = batch_correction.pca_projection(tic_std)
    X_std = batch_correction.standardize_features(matrix.X)
    msm_pca, msm_evr = batch_correction.pca_projection(X_std)
    viz.save_figure(
        viz.plot_pca_tic_vs_msm(tic_pca, tic_summary["cancer_binary"].to_numpy(), tic_evr, msm_pca, matrix.y, msm_evr),
        paths.figures / "06_pca_tic_vs_msm_features.png",
    )

    X_combat = batch_correction.apply_combat_correction(X_std, matrix.batch, matrix.y, matrix.sample_ids)
    n_nan = int(np.isnan(X_combat).sum())
    if n_nan:
        logger.warning("ComBat output contains %d NaN values; they are zero-filled for the PCA figure only.", n_nan)
    before, before_evr = batch_correction.pca_projection(X_std, random_state=PCA_RANDOM_STATE)
    after, after_evr = batch_correction.pca_projection(np.nan_to_num(X_combat), random_state=PCA_RANDOM_STATE)
    viz.save_figure(
        viz.plot_pca_before_after(before, before_evr, after, after_evr, matrix.batch, "hospital"),
        paths.figures / "07_pca_before_after_combat_by_hospital.png",
    )
    viz.save_figure(
        viz.plot_pca_before_after(
            before, before_evr, after, after_evr, matrix.y, "cancer label", names=viz.LABEL_NAMES
        ),
        paths.figures / "08_pca_before_after_combat_by_label.png",
    )
    logger.info(
        "PCA explained variance (PC1, PC2): TIC summary %s | features %s | after ComBat %s",
        np.round(tic_evr, 4),
        np.round(msm_evr, 4),
        np.round(after_evr, 4),
    )
    summary = {
        "n_labelled_samples": len(matrix.y),
        "n_model_features": matrix.numeric.shape[1],
        "model_feature_columns": list(matrix.numeric.columns),
        "tic_summary_pc12_explained_variance": tic_evr,
        "msm_pc12_explained_variance": msm_evr,
        "after_combat_pc12_explained_variance": after_evr,
        "combat_output_nan_values": n_nan,
    }
    return X_combat, summary


def build_image_records(
    tic: pd.DataFrame, dynamics: dict[str, SampleDynamics], matrix: ModelMatrix, image_dir: Path, settings: Settings
) -> list[dict]:
    """Save a TIC image for every labelled sample and return dataset records aligned with ``matrix`` rows."""
    records = []
    for row, (sample_id, label) in enumerate(zip(matrix.sample_ids, matrix.y, strict=True)):
        path = save_tic_image(
            build_sample_tic_image(sample_id, tic, dynamics, settings), image_dir / f"{sample_id}.png"
        )
        records.append({"sample_id": sample_id, "path": str(path), "row": row, "label": int(label)})
    return records


def run_deep_model_stage(
    tic: pd.DataFrame,
    dynamics: dict[str, SampleDynamics],
    matrix: ModelMatrix,
    X_combat: np.ndarray,
    paths: OutputPaths,
    settings: Settings,
) -> dict:
    """Train and compare the three neural models, then time their forward passes; figures 09-10 (Steps 16-19)."""
    from msm_olfactory import deep_models

    records = build_image_records(tic, dynamics, matrix, paths.cache / "tic_images", settings)
    train_idx, val_idx = stratified_holdout_indices(matrix.y, settings)
    comparison = deep_models.train_and_compare_models(
        records, train_idx, val_idx, X_combat, settings, torch_seed=settings.base_seed
    )
    runs, reference = comparison.runs, comparison.always_cancer_val_f1
    n_val = len(comparison.val_records)
    logger.info("  Reference: always predicting Cancer gives validation F1 %.3f", reference)

    history = pd.DataFrame([{"model": name, **h} for name, run in runs.items() for h in run.history])
    history.to_csv(paths.results / "deep_model_training_history.csv", index=False)
    viz.save_figure(
        viz.plot_training_curves({name: run.history for name, run in runs.items()}, n_val, reference),
        paths.figures / "09_deep_model_validation_curves.png",
    )
    viz.save_figure(
        viz.plot_model_comparison(
            {name: run.best_val_f1 for name, run in runs.items()},
            f"Best validation F1 over {settings.epochs} epochs (selected on the validation split)",
            n_val,
            reference,
        ),
        paths.figures / "10_deep_model_comparison_f1.png",
    )

    cnn, mlp = comparison.models[CNN_NAME], comparison.models[MLP_NAME]
    t_cnn = deep_models.time_forward_pass_on_loader(cnn, comparison.val_loader)
    t_mlp = deep_models.time_forward_pass_on_loader(mlp, comparison.val_loader)
    overhead = {
        "cnn_seconds_per_batch": t_cnn,
        "mlp_seconds_per_batch": t_mlp,
        "speedup_cnn_over_mlp": t_cnn / t_mlp,
        "cnn_parameters": deep_models.count_parameters(cnn),
        "mlp_parameters": deep_models.count_parameters(mlp),
        "multimodal_parameters": deep_models.count_parameters(comparison.models[MULTIMODAL_NAME]),
        "cnn_inputs_per_sample": N_IMAGE_ROWS * settings.image_width,
        "mlp_inputs_per_sample": X_combat.shape[1],
        "note": "Forward pass only. Excludes MSM fitting and rolling ADF/KPSS, which the MLP's inputs and "
        "the image's rows 1-3 both require.",
    }
    return {
        "n_train": len(comparison.train_records),
        "n_val": n_val,
        "val_class_counts": {
            0: sum(r["label"] == 0 for r in comparison.val_records),
            1: sum(r["label"] == 1 for r in comparison.val_records),
        },
        "always_cancer_val_f1": reference,
        "best_val_f1": {name: run.best_val_f1 for name, run in runs.items()},
        "best_epoch": {name: run.best_epoch for name, run in runs.items()},
        "final_epoch_val_f1": {name: run.final_val_f1 for name, run in runs.items()},
        "computational_overhead": overhead,
    }


# --- entry point --------------------------------------------------------------------------------------------------


def run_pipeline(
    data_root: Path, output_dir: Path, settings: Settings = DEFAULT_SETTINGS, is_synthetic: bool = False
) -> dict:
    """Run the full analysis and write figures, aggregate results and ``results/run_summary.json``.

    Parameters
    ----------
    data_root
        Folder with the extracted dataset (``Samples Demographics.csv`` and the site folders).
    output_dir
        Output root; ``figures/``, ``results/`` and ``cache/`` are created below it.
    settings
        Analysis settings (defaults reproduce the source).
    is_synthetic
        Recorded in the summary so synthetic demo results cannot be mistaken for real ones.

    Returns
    -------
    dict
        The run summary (also written to ``results/run_summary.json``).
    """
    paths = OutputPaths(Path(output_dir))
    summary: dict = {"data_root_is_synthetic": is_synthetic, "settings": asdict(settings)}

    tic = load_tic_table(Path(data_root), paths, settings)
    summary["ingestion"] = data.summarize_ingestion(tic)
    logger.info(
        "Loaded %d TICs (%s rows); %d samples have clinical labels %s",
        summary["ingestion"]["samples"],
        f"{summary['ingestion']['tic_rows']:,}",
        summary["ingestion"]["labelled_samples"],
        summary["ingestion"]["labelled_by_hospital"],
    )

    dynamics = load_sample_dynamics(tic, paths, settings)
    tables = features.build_feature_tables(tic, dynamics)
    for name, table in tables.items():
        table.to_parquet(paths.cache_file(name, settings, ".parquet"), index=False)
    failed = sorted(s for s, d in dynamics.items() if not d.ok)
    k_counts = tables["msm_features"]["k_used"].value_counts().to_dict()
    summary["msm"] = {
        "fitted_samples": len(tables["msm_features"]),
        "failed_samples": failed,
        "k_used_counts": k_counts,
    }
    logger.info(
        "MSM fitted for %d samples (regimes used: %s); failed: %d", len(tables["msm_features"]), k_counts, len(failed)
    )

    full = tables["msm_features_full"]
    summary["illustrative_figures"] = plot_illustrative_figures(paths, settings)
    summary["example_sample"] = summarize_reference_sample(dynamics, settings)
    summary["tabular_baselines"] = run_tabular_stage(full, paths, settings)
    summary["image_panel_samples"] = plot_image_panel(tic, dynamics, paths, settings)
    matrix = features.build_model_matrix(full)
    X_combat, summary["pca"] = run_batch_correction_stage(tic, matrix, paths)
    if settings.run_deep_models:
        summary["deep_models"] = run_deep_model_stage(tic, dynamics, matrix, X_combat, paths, settings)

    write_json(paths.results / "run_summary.json", summary)
    logger.info("Results written to %s", paths.results)
    return summary
