"""Evaluation protocol shared by all models: the hold-out split, classification metrics and the reference classifier."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from msm_olfactory.config import Settings

#: Probability threshold for converting scores into Cancer / Non-Cancer predictions (as in the source).
DECISION_THRESHOLD = 0.5


def stratified_holdout_indices(labels: Sequence[int], settings: Settings) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the single stratified train/validation split used by every model (source protocol).

    Uses ``train_test_split(test_size=settings.test_size, random_state=settings.random_state, stratify=labels)``.
    The tabular and deep stages therefore validate on exactly the same samples.

    Raises
    ------
    ValueError
        If a class has fewer than two samples (stratification is impossible).
    """
    labels = np.asarray(labels)
    _, counts = np.unique(labels, return_counts=True)
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError(f"Stratified split needs at least 2 samples per class; class counts are {counts.tolist()}.")
    train_idx, val_idx = train_test_split(
        np.arange(len(labels)), test_size=settings.test_size, random_state=settings.random_state, stratify=labels
    )
    return train_idx, val_idx


def evaluate_probabilities(y_true: np.ndarray, p_hat: np.ndarray, threshold: float = DECISION_THRESHOLD) -> dict:
    """Accuracy, F1, ROC-AUC, PR-AUC and confusion matrix for predicted Cancer probabilities.

    ROC-AUC and PR-AUC are NaN when ``y_true`` contains a single class.
    """
    y_pred = (p_hat >= threshold).astype(int)
    two_classes = len(np.unique(y_true)) == 2
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, p_hat)) if two_classes else float("nan"),
        "pr_auc": float(average_precision_score(y_true, p_hat)) if two_classes else float("nan"),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def always_cancer_f1(y_true: Sequence[int]) -> float:
    """F1 of the constant classifier that labels every sample Cancer.

    Equals ``2p / (1 + p)``, where p is the share of Cancer samples. It is reported next to every F1 (not in the
    source) so that scores can be read against class prevalence.
    """
    p = float(np.mean(np.asarray(y_true) == 1))
    return 2 * p / (1 + p) if p > 0 else 0.0
