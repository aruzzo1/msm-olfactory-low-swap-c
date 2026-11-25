"""Markov-switching analysis of GC-MS total ion chromatograms.

Each chromatogram is fitted with a regime-switching regression (statsmodels ``MarkovRegression``, ``trend="c"``,
``switching_variance=False``)::

    y_t = mu_{S_t} + e_t,      e_t ~ N(0, sigma^2)        one variance shared by all regimes
    P(S_t = i | S_{t-1} = j) = p_ij                        time-invariant K x K transition matrix

where ``y_t`` is the log1p + z-scored intensity and ``S_t`` is a hidden regime in {0, ..., K-1}. Because only the mean
switches, the regimes act as learned intensity bands (baseline, intermediate, peak-dominated points). Their
occupancy and transition probabilities become compact per-sample features. The model characterises signal
dynamics; it is not itself a classifier.

Orientation: statsmodels returns ``P[i, j] = P(S_t = i | S_{t-1} = j)``, so each COLUMN sums to 1. The feature
``trans_i_j`` keeps that orientation (probability of moving TO regime i FROM regime j).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from msm_olfactory.config import MAX_REGIMES_IN_FEATURES, Settings
from msm_olfactory.preprocessing import log_zscore_transform


@dataclass
class MsmFit:
    """Result of fitting the MSM to one chromatogram.

    Attributes
    ----------
    sample_id, n_points, k_used
        Sample, series length and number of regimes actually fitted (3, or 2 after a fallback).
    llf, aic, bic
        Log-likelihood and information criteria of the fit.
    regime_probs
        ``(T, K)`` smoothed marginal probabilities ``P(S_t = k | y_1..T)``.
    regimes
        ``(T,)`` hard regime labels (argmax of ``regime_probs``). Labels are arbitrary per fit (label switching).
    transition
        ``(K, K)`` column-stochastic transition matrix.
    """

    sample_id: str
    n_points: int
    k_used: int
    llf: float
    aic: float
    bic: float
    regime_probs: np.ndarray
    regimes: np.ndarray
    transition: np.ndarray


def _fit_markov_regression(y: np.ndarray, k_regimes: int, settings: Settings, seed: int | None):
    """One MarkovRegression fit with the source's EM and random-start search settings."""
    model = MarkovRegression(y, k_regimes=k_regimes, trend="c", switching_variance=False)
    result = model.fit(
        em_iter=settings.em_iter,
        search_reps=settings.search_reps,
        search_iter=settings.search_iter,
        disp=False,
        rng=seed,
    )
    return model, result


def smoothed_probabilities(result, k_used: int, nobs: int) -> np.ndarray:
    """Smoothed regime probabilities as a ``(T, K)`` array, whichever orientation statsmodels returns.

    Raises
    ------
    RuntimeError
        If the array has neither shape ``(T, K)`` nor ``(K, T)``.
    """
    probs = np.asarray(result.smoothed_marginal_probabilities)
    if probs.shape == (nobs, k_used):
        return probs
    if probs.shape == (k_used, nobs):
        return probs.T
    raise RuntimeError(f"Unexpected smoothed probability shape {probs.shape}; expected ({nobs}, {k_used}).")


def transition_matrix(model, result) -> np.ndarray:
    """Time-invariant ``(K, K)`` transition matrix from the fitted parameters (columns sum to 1)."""
    trans = np.asarray(model.regime_transition_matrix(result.params))
    if trans.ndim == 3:
        trans = trans[:, :, 0] if trans.shape[2] == 1 else trans.mean(axis=2)
    return trans


def fit_markov_switching_model(sample_id: str, intensity: np.ndarray, settings: Settings, seed: int | None) -> MsmFit:
    """Fit the Markov switching model to one total ion chromatogram.

    Parameters
    ----------
    sample_id
        Identifier stored in the result.
    intensity
        Raw TIC intensities in retention-time order. They are log1p + z-score transformed before fitting.
    settings
        Uses ``k_regimes``, ``fallback_k_regimes``, ``msm_min_points`` and the EM/search settings.
    seed
        Seed for the random start-value search. ``None`` reproduces the source's unseeded behaviour (not
        reproducible between runs).

    Returns
    -------
    MsmFit

    Raises
    ------
    ValueError
        If the series is shorter than ``settings.msm_min_points``, or if both the 3-regime fit and the 2-regime
        fallback raise ``ValueError``.

    Notes
    -----
    The model characterises latent signal regimes. It does not provide a diagnostic classification by itself.
    """
    if len(intensity) < settings.msm_min_points:
        raise ValueError(f"{sample_id}: {len(intensity)} points; need at least {settings.msm_min_points} for the MSM.")
    y = log_zscore_transform(intensity)
    try:
        model, result = _fit_markov_regression(y, settings.k_regimes, settings, seed)
        k_used = settings.k_regimes
    except ValueError:
        # The source falls back to fewer regimes when the K-regime fit fails (e.g. a degenerate regime).
        model, result = _fit_markov_regression(y, settings.fallback_k_regimes, settings, seed)
        k_used = settings.fallback_k_regimes
    probs = smoothed_probabilities(result, k_used, len(y))
    return MsmFit(
        sample_id=sample_id,
        n_points=len(intensity),
        k_used=k_used,
        llf=float(result.llf),
        aic=float(result.aic),
        bic=float(result.bic),
        regime_probs=probs,
        regimes=probs.argmax(axis=1),
        transition=transition_matrix(model, result),
    )


def extract_msm_features(fit: MsmFit) -> dict:
    """Convert an MSM fit into flat per-sample features (source Step 9).

    Returns
    -------
    dict
        ``sample_id, n_points, k_used, llf, aic, bic``, ``regime_frac_{0,1,2}`` (share of points in each regime) and
        ``trans_{i}_{j}`` for i, j in {0, 1, 2}. Entries for regimes that were not fitted are NaN.

    Notes
    -----
    Regime numbers are arbitrary in each fit (label switching), so ``regime_frac_r`` and ``trans_i_j`` for a given
    r, i, j do not refer to the same physical regime across samples. See ``docs/methodology.md``.
    """
    counts = np.bincount(fit.regimes, minlength=fit.k_used)
    fractions = counts / len(fit.regimes)
    row = {
        "sample_id": fit.sample_id,
        "n_points": fit.n_points,
        "k_used": fit.k_used,
        "llf": fit.llf,
        "aic": fit.aic,
        "bic": fit.bic,
    }
    for r in range(MAX_REGIMES_IN_FEATURES):
        row[f"regime_frac_{r}"] = fractions[r] if r < fit.k_used else np.nan
    for i in range(MAX_REGIMES_IN_FEATURES):
        for j in range(MAX_REGIMES_IN_FEATURES):
            fitted = i < fit.k_used and j < fit.k_used
            row[f"trans_{i}_{j}"] = fit.transition[i, j] if fitted else np.nan
    return row
