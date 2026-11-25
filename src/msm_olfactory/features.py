"""Assemble per-sample feature tables and the matrices used by the models.

Three views of the same features are built, as in the source:

* ``msm_features``          17 MSM statistics per sample (Step 9)
* ``msm_features_labeled``  plus clinical labels (Step 9a)
* ``msm_features_full``     plus 11 stationarity statistics (Step 11a)

From ``msm_features_full`` two model inputs are derived:

* :func:`build_model_matrix`: every numeric column except the label (30 columns) for PCA, ComBat and the deep models
  (Steps 13-18).
* :func:`prepare_tabular_features`: the same columns minus constant ones, for the logistic regression and random
  forest baselines (Step 11b).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from msm_olfactory import data
from msm_olfactory.sample_dynamics import SampleDynamics

#: A feature whose name contains any of these tokens is grouped as "Stationarity"; every other feature is grouped as
#: "MSM". By this rule the clinical covariates age and psa fall into the "MSM" group, as in the source.
STATIONARITY_TOKENS = (
    "adf",
    "kpss",
    "both_p",
    "frac_both",
    "mean_adf",
    "median_adf",
    "mean_kpss",
    "median_kpss",
    "frac_adf",
    "frac_kpss",
    "station",
)
TIC_SUMMARY_COLUMNS = ["tic_mean", "tic_std", "tic_max", "tic_median"]


def build_feature_tables(tic: pd.DataFrame, dynamics: dict[str, SampleDynamics]) -> dict[str, pd.DataFrame]:
    """Build the three per-sample feature tables from the per-sample dynamics.

    Samples whose MSM failed are omitted, as in the source.

    Returns
    -------
    dict
        ``{"msm_features", "msm_features_labeled", "msm_features_full"}`` DataFrames, one row per sample.
    """
    fitted = [d for d in dynamics.values() if d.ok]
    msm_features = pd.DataFrame([d.msm_features for d in fitted])
    labeled = msm_features.merge(data.get_sample_labels(tic), on="sample_id", how="left")
    station = pd.DataFrame([d.stationarity_features for d in fitted if d.stationarity_features is not None])
    full = labeled.merge(station, on="sample_id", how="left") if not station.empty else labeled
    return {"msm_features": msm_features, "msm_features_labeled": labeled, "msm_features_full": full}


@dataclass
class ModelMatrix:
    """Labelled modelling matrix used for PCA, ComBat and the deep models.

    Attributes
    ----------
    frame
        Labelled rows of the feature table.
    numeric
        Feature columns used as model inputs.
    y
        0/1 cancer labels aligned with ``numeric``.
    batch
        Hospital of each sample (the ComBat batch variable).
    sample_ids
        Sample IDs aligned with ``numeric``.
    """

    frame: pd.DataFrame
    numeric: pd.DataFrame
    y: np.ndarray
    batch: np.ndarray
    sample_ids: np.ndarray

    @property
    def X(self) -> np.ndarray:  # noqa: N802 - conventional name for a design matrix
        """Feature values as a float array, shape ``(n_samples, n_features)``."""
        return self.numeric.to_numpy(dtype=float)


def _labelled_rows(features_full: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Rows with a 0/1 ``cancer_binary`` label, and the numeric label column."""
    target = pd.to_numeric(features_full["cancer_binary"], errors="coerce")
    mask = target.isin([0, 1])
    return features_full.loc[mask].copy().reset_index(drop=True), target[mask].astype(int).reset_index(drop=True)


def build_model_matrix(features_full: pd.DataFrame) -> ModelMatrix:
    """Select labelled samples and use every numeric column except the label as a model input.

    This reproduces the source exactly, so the 30 inputs include the clinical covariates ``age`` and ``psa`` and the
    constant columns ``n_points``, ``k_used`` and ``n_station_windows`` (see ``docs/methodology.md``).

    Raises
    ------
    KeyError
        If ``sample_id``, ``hospital`` or ``cancer_binary`` is missing.
    ValueError
        If there are no numeric features, or the labelled samples do not include both classes.
    """
    for column in ("sample_id", "hospital", "cancer_binary"):
        if column not in features_full.columns:
            raise KeyError(f"Feature table is missing required column {column!r}")
    frame, y = _labelled_rows(features_full)
    if y.nunique() < 2:
        raise ValueError(f"Need labelled samples from both classes; found classes {sorted(y.unique())}.")
    frame["hospital"] = frame["hospital"].astype(str).str.strip()
    numeric = frame.select_dtypes(include=[np.number]).drop(columns=["cancer_binary"], errors="ignore")
    if numeric.shape[1] == 0:
        raise ValueError("No numeric feature columns found.")
    return ModelMatrix(
        frame=frame,
        numeric=numeric,
        y=y.to_numpy(),
        batch=frame["hospital"].to_numpy(),
        sample_ids=frame["sample_id"].astype(str).to_numpy(),
    )


@dataclass
class TabularFeatures:
    """Inputs for the tabular baselines.

    Attributes
    ----------
    X
        Non-constant numeric features with missing values filled by the column median.
    y
        0/1 labels.
    msm_features, stationarity_features
        Column names in each group (``age`` and ``psa`` fall in the MSM group).
    dropped_constant
        Columns removed because they have a single value.
    """

    X: pd.DataFrame
    y: np.ndarray
    msm_features: list[str]
    stationarity_features: list[str]
    dropped_constant: list[str]


def prepare_tabular_features(features_full: pd.DataFrame) -> TabularFeatures:
    """Prepare the tabular-baseline inputs (source Step 11b).

    Keeps labelled rows and numeric columns, drops all-NaN and constant columns, and fills remaining NaN with the
    column median.

    Raises
    ------
    KeyError
        If neither ``cancer_binary`` nor ``cancer_status`` is present.
    """
    df = features_full.copy()
    if "cancer_binary" in df.columns:
        df["cancer_binary"] = pd.to_numeric(df["cancer_binary"], errors="coerce")
    elif "cancer_status" in df.columns:
        # The source's fallback used str.contains("cancer"), which would also mark "Non-Cancer" as 1. That branch never
        # ran (cancer_binary is always present); this implementation uses the correct rule instead.
        df["cancer_binary"] = df.apply(data.cancer_flag, axis=1).astype(float)
    else:
        raise KeyError("No cancer label found (need 'cancer_binary' or 'cancer_status').")
    df = df[df["cancer_binary"].isin([0, 1])].copy()

    X = df.select_dtypes(include=[np.number]).drop(columns=["cancer_binary"])
    X = X.drop(columns=[c for c in X.columns if X[c].isna().all()])
    constant = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    X = X.drop(columns=constant).apply(lambda s: s.fillna(s.median()), axis=0).reset_index(drop=True)
    stationarity = [c for c in X.columns if any(token in c.lower() for token in STATIONARITY_TOKENS)]
    msm = [c for c in X.columns if c not in stationarity]
    return TabularFeatures(
        X=X,
        y=df["cancer_binary"].astype(int).to_numpy(),
        msm_features=msm,
        stationarity_features=stationarity,
        dropped_constant=constant,
    )


def tic_summary_statistics(tic: pd.DataFrame) -> pd.DataFrame:
    """Per-sample mean, std, max and median intensity with hospital and label; unlabelled samples dropped (Step 13)."""
    summary = tic.groupby("sample_id")["intensity"].agg(["mean", "std", "max", "median"])
    summary.columns = TIC_SUMMARY_COLUMNS
    meta = tic.groupby("sample_id").agg(hospital=("hospital", "first"), cancer_binary=("cancer_binary", "first"))
    out = summary.join(meta).reset_index().dropna(subset=["cancer_binary"])
    out["cancer_binary"] = out["cancer_binary"].astype(int)
    return out
