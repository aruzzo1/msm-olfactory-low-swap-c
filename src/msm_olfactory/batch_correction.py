"""Feature standardisation, empirical Bayes (ComBat) batch correction, and PCA projections for inspecting batches.

ComBat (Johnson et al. 2007, via ``neuroCombat``) models each standardized feature g of sample j in batch i as::

    x_ijg = alpha_g + X_j beta_g + gamma_ig + delta_ig * e_ijg

It removes the batch location (gamma) and scale (delta) effects, shrinking them towards common priors with empirical
Bayes. Here the batch is the hospital, and ``cancer_binary`` is a protected covariate (X_j).

Leakage caveat, reproduced from the source: the scaler and ComBat are fitted on ALL labelled samples, with the outcome
as the protected covariate, before the train/validation split used by the deep models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from msm_olfactory.utils import quiet_numerics, suppress_stdout


def standardize_features(X: np.ndarray) -> np.ndarray:
    """Z-score each column (scikit-learn ``StandardScaler`` fitted on ``X`` itself)."""
    return StandardScaler().fit_transform(X)


def apply_combat_correction(
    X_std: np.ndarray, batch: np.ndarray, labels: np.ndarray, sample_ids: np.ndarray
) -> np.ndarray:
    """Remove hospital batch effects from standardized features with parametric ComBat.

    Parameters
    ----------
    X_std
        Standardized features, shape ``(n_samples, n_features)``.
    batch
        Hospital of each sample.
    labels
        0/1 cancer labels, passed as a categorical covariate to preserve.
    sample_ids
        Row identifiers (used as the covariate index).

    Returns
    -------
    numpy.ndarray
        Harmonized features, shape ``(n_samples, n_features)``.

    Raises
    ------
    ValueError
        If there are fewer than two batches, or the inputs are not aligned.
    """
    from neuroCombat import neuroCombat  # imported lazily: optional at import time, and slow

    if len(np.unique(batch)) < 2:
        raise ValueError("ComBat needs at least two batches (hospitals).")
    if not (len(X_std) == len(batch) == len(labels) == len(sample_ids)):
        raise ValueError("X_std, batch, labels and sample_ids must have the same number of samples.")
    # Hospital is the batch variable, so site-specific shifts are removed. The label is passed as a protected
    # covariate so that ComBat does not remove the cancer-related variation along with the site effect.
    covars = pd.DataFrame({"batch": batch, "cancer_binary": labels}, index=sample_ids)
    # neuroCombat prints five progress lines per call, and warns about division by zero on the constant feature
    # columns (n_points, k_used, n_station_windows). Those columns stay zero after standardisation; the output is
    # checked for NaN by the caller.
    with suppress_stdout(), quiet_numerics():
        result = neuroCombat(
            dat=X_std.T,  # neuroCombat expects (features, samples)
            covars=covars,
            batch_col="batch",
            categorical_cols=["cancer_binary"],
            continuous_cols=None,
            parametric=True,
        )
    return np.asarray(result["data"]).T


def pca_projection(X: np.ndarray, random_state: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Project onto the first two principal components.

    Returns
    -------
    tuple
        ``(coords, explained_variance_ratio)``: ``coords`` has shape ``(n_samples, 2)``; the ratios cover PC1 and
        PC2.
    """
    pca = PCA(n_components=2, random_state=random_state)
    return pca.fit_transform(X), pca.explained_variance_ratio_[:2]
