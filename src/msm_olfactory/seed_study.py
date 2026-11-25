"""Seed-sensitivity study: the source protocol repeated under several MSM and PyTorch seeds (not in the source).

The source notebook was unseeded, and its technical report quotes F1 values from six runs. This study measures how
far the same pipeline's scores move when only the random seeds change. Data, split, features, ComBat settings,
architectures and training settings are unchanged.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pandas as pd

from msm_olfactory import batch_correction, features, tabular_models
from msm_olfactory.config import Settings
from msm_olfactory.evaluation import stratified_holdout_indices
from msm_olfactory.pipeline import OutputPaths, build_image_records, load_sample_dynamics, load_tic_table

logger = logging.getLogger(__name__)

TABULAR_COLUMNS = {
    "logreg_msm_group": "logreg_msm_group",
    "logreg_all_features": "logreg_all",
    "random_forest_all_features": "random_forest",
}


def run_seed_study(
    data_root: Path, output_dir: Path, settings: Settings, msm_seeds: Sequence[int], torch_seeds: Sequence[int]
) -> pd.DataFrame:
    """Repeat the analysis for every (MSM seed, PyTorch seed) pair.

    For each MSM seed the per-sample fits, features, tabular baselines and ComBat matrix are recomputed (fits are
    cached per seed). For each PyTorch seed the three neural models are retrained on the same split.

    Returns
    -------
    pandas.DataFrame
        One row per (MSM seed, PyTorch seed) with tabular F1/ROC-AUC, each model's best and final-epoch validation F1
        and best epoch, and the always-Cancer reference F1. Also written to ``results/seed_study.csv``.
    """
    from msm_olfactory import deep_models

    paths = OutputPaths(Path(output_dir))
    tic = load_tic_table(Path(data_root), paths, settings)
    rows = []
    for msm_seed in msm_seeds:
        seeded = replace(settings, base_seed=msm_seed)
        dynamics = load_sample_dynamics(tic, paths, seeded)
        full = features.build_feature_tables(tic, dynamics)["msm_features_full"]
        tabular = tabular_models.run_tabular_baselines(features.prepare_tabular_features(full), seeded).metrics
        matrix = features.build_model_matrix(full)
        X_combat = batch_correction.apply_combat_correction(
            batch_correction.standardize_features(matrix.X), matrix.batch, matrix.y, matrix.sample_ids
        )
        records = build_image_records(tic, dynamics, matrix, paths.cache / f"tic_images_seed{msm_seed}", seeded)
        train_idx, val_idx = stratified_holdout_indices(matrix.y, seeded)
        for torch_seed in torch_seeds:
            comparison = deep_models.train_and_compare_models(records, train_idx, val_idx, X_combat, seeded, torch_seed)
            row = {
                "msm_seed": msm_seed,
                "torch_seed": torch_seed,
                "always_cancer_val_f1": comparison.always_cancer_val_f1,
            }
            for key, prefix in TABULAR_COLUMNS.items():
                row[f"{prefix}_f1"] = tabular[key]["f1"]
                row[f"{prefix}_roc_auc"] = tabular[key]["roc_auc"]
            for name, run in comparison.runs.items():
                row[f"{name} | best_val_f1"] = run.best_val_f1
                row[f"{name} | best_epoch"] = run.best_epoch
                row[f"{name} | final_val_f1"] = run.final_val_f1
            rows.append(row)
    table = pd.DataFrame(rows)
    paths.results.mkdir(parents=True, exist_ok=True)
    table.to_csv(paths.results / "seed_study.csv", index=False)
    logger.info("Seed study: %d runs written to %s", len(table), paths.results / "seed_study.csv")
    return table
