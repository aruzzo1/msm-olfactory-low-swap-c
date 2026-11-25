"""Tabular baselines on the MSM + stationarity features (source Step 11b).

Logistic regression is fitted on standardized features (scaler fitted on the training split only), once on the
"MSM-group" columns and once on all columns. A random forest is fitted on unscaled features and supplies feature
importances. The source's markdown also mentions temperature and Platt calibration, but no calibration code exists in
the v18 notebook, so none is implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from msm_olfactory.config import Settings
from msm_olfactory.evaluation import always_cancer_f1, evaluate_probabilities, stratified_holdout_indices
from msm_olfactory.features import TabularFeatures


@dataclass
class TabularResults:
    """Metrics of the three baselines, plus logistic-regression coefficients and random-forest importances."""

    metrics: dict
    coefficients: pd.DataFrame
    importance: pd.DataFrame


def _logistic_regression(settings: Settings) -> LogisticRegression:
    # Class-balanced weights offset the 38:27 Cancer:Non-Cancer imbalance; liblinear suits small samples.
    return LogisticRegression(
        max_iter=2000, class_weight="balanced", solver="liblinear", random_state=settings.random_state
    )


def fit_logistic_regression(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, settings: Settings
) -> tuple[LogisticRegression, dict]:
    """Fit a class-balanced logistic regression on standardized features and evaluate it on the validation split."""
    model = _logistic_regression(settings).fit(X_train, y_train)
    return model, evaluate_probabilities(y_val, model.predict_proba(X_val)[:, 1])


def fit_random_forest(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, settings: Settings
) -> tuple[RandomForestClassifier, dict]:
    """Fit a class-balanced random forest (unscaled features) and evaluate it on the validation split."""
    model = RandomForestClassifier(
        n_estimators=settings.rf_n_estimators, random_state=settings.random_state, class_weight="balanced", n_jobs=-1
    ).fit(X_train, y_train)
    return model, evaluate_probabilities(y_val, model.predict_proba(X_val)[:, 1])


def run_tabular_baselines(features: TabularFeatures, settings: Settings) -> TabularResults:
    """Train and evaluate the three tabular baselines on the shared stratified hold-out split.

    Returns
    -------
    TabularResults
        ``metrics`` has the keys ``logreg_msm_group``, ``logreg_all_features`` and ``random_forest_all_features``,
        plus split sizes, class counts and the always-Cancer reference F1.
    """
    train_idx, val_idx = stratified_holdout_indices(features.y, settings)
    X = features.X.to_numpy(dtype=float)
    X_train, X_val, y_train, y_val = X[train_idx], X[val_idx], features.y[train_idx], features.y[val_idx]
    scaler = StandardScaler().fit(X_train)  # fitted on the training split only
    X_train_std, X_val_std = scaler.transform(X_train), scaler.transform(X_val)
    names = list(features.X.columns)

    metrics: dict = {
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "always_cancer_val_f1": always_cancer_f1(y_val),
        "class_counts": {int(k): int(v) for k, v in zip(*np.unique(features.y, return_counts=True), strict=True)},
        "n_msm_group_features": len(features.msm_features),
        "n_stationarity_features": len(features.stationarity_features),
        "dropped_constant_features": features.dropped_constant,
    }
    if features.msm_features:
        cols = [names.index(c) for c in features.msm_features]
        _, metrics["logreg_msm_group"] = fit_logistic_regression(
            X_train_std[:, cols], y_train, X_val_std[:, cols], y_val, settings
        )
    lr_all, metrics["logreg_all_features"] = fit_logistic_regression(X_train_std, y_train, X_val_std, y_val, settings)
    forest, metrics["random_forest_all_features"] = fit_random_forest(X_train, y_train, X_val, y_val, settings)

    coefficients = (
        pd.DataFrame({"feature": names, "coef": lr_all.coef_.ravel()})
        .sort_values("coef", ascending=False)
        .reset_index(drop=True)
    )
    importance = (
        pd.DataFrame({"feature": names, "importance": forest.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance["group"] = np.where(importance["feature"].isin(features.stationarity_features), "Stationarity", "MSM")
    return TabularResults(metrics=metrics, coefficients=coefficients, importance=importance)
