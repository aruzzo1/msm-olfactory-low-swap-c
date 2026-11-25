"""Markov-switching and stationarity features of GC-MS chromatograms: a SWaP-C proof of concept.

Reconstructed, reproducible implementation of an applied research notebook built on the public GC-MS dataset of
Rotteveel et al. (2025). The main entry points are exported here::

    from msm_olfactory import Settings, run_pipeline
    summary = run_pipeline("data/raw/rotteveel_data", "outputs", Settings(workers=8))

Names that need PyTorch (the neural models and the pipeline) are imported lazily on first use. Lightweight work, such
as the parallel per-sample MSM fits, therefore never loads PyTorch.
"""

from __future__ import annotations

import importlib
from typing import Any

from msm_olfactory.config import Settings
from msm_olfactory.data import build_tic_table, load_metadata, load_tic_data, merge_metadata
from msm_olfactory.images import build_tic_image
from msm_olfactory.msm import MsmFit, extract_msm_features, fit_markov_switching_model
from msm_olfactory.preprocessing import log_zscore_transform
from msm_olfactory.stationarity import compute_stationarity_features, rolling_stationarity_tests

__version__ = "1.0.0"

_LAZY = {
    "apply_combat_correction": "msm_olfactory.batch_correction",
    "build_feature_tables": "msm_olfactory.features",
    "build_model_matrix": "msm_olfactory.features",
    "run_tabular_baselines": "msm_olfactory.tabular_models",
    "train_and_compare_models": "msm_olfactory.deep_models",
    "TicOnlyCNN": "msm_olfactory.deep_models",
    "MsmOnlyMLP": "msm_olfactory.deep_models",
    "MultiModalResNet": "msm_olfactory.deep_models",
    "run_pipeline": "msm_olfactory.pipeline",
    "run_seed_study": "msm_olfactory.seed_study",
}

__all__ = [
    "MsmFit",
    "Settings",
    "build_tic_image",
    "build_tic_table",
    "compute_stationarity_features",
    "extract_msm_features",
    "fit_markov_switching_model",
    "load_metadata",
    "load_tic_data",
    "log_zscore_transform",
    "merge_metadata",
    "rolling_stationarity_tests",
    *_LAZY,
]


def __getattr__(name: str) -> Any:
    """Import heavy public names on first access (PEP 562)."""
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'msm_olfactory' has no attribute {name!r}")
