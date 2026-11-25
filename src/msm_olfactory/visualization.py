"""Figures for the analysis. Every function returns a matplotlib Figure and never alters the data it plots.

Styling: one categorical palette in fixed order (blue, orange, aqua), a recessive grey grid, and a second encoding
(marker shape, line style or hatching) wherever colour distinguishes series. Figures therefore stay readable in
greyscale print and for colour-vision-deficient readers. The source used matplotlib defaults and forced green/red for
the two hospitals; only colours differ here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; the notebook switches to its inline backend itself
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from msm_olfactory.config import (
    CNN_NAME,
    IMAGE_ROW_LABELS,
    MLP_NAME,
    MODEL_NAMES,
    MULTIMODAL_NAME,
    N_IMAGE_ROWS,
    P_THRESHOLD,
)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]
LINESTYLES = ["-", "--", ":"]
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#dcdbd5", "#fcfcfb"
LABEL_NAMES = {0: "Non-Cancer", 1: "Cancer"}
SHORT_MODEL_NAMES = {CNN_NAME: "TIC-image CNN", MLP_NAME: "MSM-feature MLP", MULTIMODAL_NAME: "Multi-modal ResNet18"}

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "legend.frameon": False,
        "savefig.facecolor": SURFACE,
    }
)


def save_figure(fig: Figure, path: Path) -> Path:
    """Save ``fig`` as an image (format from the extension), close it, and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --- single-sample figures ----------------------------------------------------------------------------------------


def _regime_scatter(ax, time: np.ndarray, intensity: np.ndarray, regimes: np.ndarray, k_used: int) -> None:
    ax.plot(time, intensity, color=GRID, linewidth=0.8, zorder=1)
    for r in range(k_used):
        in_regime = regimes == r
        ax.scatter(
            time[in_regime],
            intensity[in_regime],
            s=6,
            color=SERIES[r % 3],
            marker=MARKERS[r % 3],
            linewidths=0,
            label=f"Regime {r} ({100 * in_regime.mean():.1f}% of points)",
            zorder=2,
        )
    ax.legend(title="MSM regime (smoothed argmax)", loc="upper right", markerscale=2.5)
    ax.set_ylabel("TIC intensity (raw counts)")


def plot_msm_regimes(
    time: np.ndarray, intensity: np.ndarray, regimes: np.ndarray, k_used: int, sample_label: str
) -> Figure:
    """One TIC with each point coloured by its MSM regime (source Step 8).

    ``sample_label`` is a display name; published figures use an anonymous label rather than the sample ID.
    """
    fig, ax = plt.subplots(figsize=(11, 4.2))
    _regime_scatter(ax, np.asarray(time), np.asarray(intensity), np.asarray(regimes), k_used)
    ax.set_xlabel("Retention time (min)")
    ax.set_title(f"{sample_label}: TIC with {k_used}-regime Markov switching model")
    return fig


def plot_msm_stationarity_panel(
    time: np.ndarray,
    intensity: np.ndarray,
    regimes: np.ndarray,
    k_used: int,
    window_times: np.ndarray,
    adf_p: np.ndarray,
    kpss_p: np.ndarray,
    sample_label: str,
    window_size: int,
    step: int,
) -> Figure:
    """MSM regimes, rolling ADF p-values and rolling KPSS p-values on a shared time axis (source Step 10)."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    _regime_scatter(axes[0], np.asarray(time), np.asarray(intensity), np.asarray(regimes), k_used)
    axes[0].set_title(f"{sample_label}: TIC with MSM regimes (k = {k_used})")
    panels = (
        (axes[1], adf_p, "ADF", "low p: window is stationary", 1.0),
        (axes[2], kpss_p, "KPSS", "low p: window is non-stationary", 0.11),  # KPSS p is clipped to [0.01, 0.10]
    )
    for ax, p_values, name, meaning, y_max in panels:
        ax.plot(window_times, p_values, color=SERIES[0], linewidth=1.2, marker="o", markersize=2.5)
        ax.axhline(P_THRESHOLD, color=SERIES[1], linestyle="--", linewidth=1.2, label=f"p = {P_THRESHOLD}")
        ax.set_ylabel(f"{name} p-value")
        ax.set_ylim(0, y_max)
        ax.set_title(f"Rolling {name} test, {window_size}-point windows every {step} points ({meaning})")
        ax.legend(loc="upper right")
    axes[2].set_xlabel("Retention time (min)")
    fig.tight_layout()
    return fig


def _show_image(ax, image: np.ndarray, title: str, show_labels: bool = True) -> None:
    ax.imshow(image, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=255, interpolation="nearest")
    ax.set_title(title)
    ax.grid(False)
    if show_labels:
        ax.set_yticks(range(N_IMAGE_ROWS), IMAGE_ROW_LABELS)
    else:
        ax.set_yticks([])


def plot_tic_image(image: np.ndarray, sample_label: str, class_label: str | None = None) -> Figure:
    """One multichannel TIC image (source Step 12b)."""
    fig, ax = plt.subplots(figsize=(9, 3.2))
    title = f"TIC image: {sample_label}" + (f" (label: {class_label})" if class_label else "")
    _show_image(ax, image, title)
    ax.set_xlabel(f"Time bin ({image.shape[1]} bins, linear resampling of the retention-time axis)")
    return fig


def plot_tic_image_panel(cancer: Mapping[str, np.ndarray], noncancer: Mapping[str, np.ndarray]) -> Figure:
    """TIC images of Cancer samples (top row) and Non-Cancer samples (bottom row) (source Step 12c).

    Dictionary keys are used as panel titles, so pass anonymous display labels rather than sample IDs.
    """
    n_cols = max(len(cancer), len(noncancer))
    fig, axes = plt.subplots(2, n_cols, figsize=(3.6 * n_cols, 5.2), squeeze=False)
    for row, (group, group_images) in enumerate((("Cancer", cancer), ("Non-Cancer", noncancer))):
        items = list(group_images.items())
        for col in range(n_cols):
            ax = axes[row, col]
            if col < len(items):
                sample_label, image = items[col]
                _show_image(ax, image, f"{sample_label}\n{group}", show_labels=(col == 0))
            else:
                ax.axis("off")
            if row == 1:
                ax.set_xlabel("Time bin")
    fig.suptitle("TIC images: Cancer (top) vs Non-Cancer (bottom)")
    fig.tight_layout()
    return fig


# --- feature-space figures ----------------------------------------------------------------------------------------


def plot_feature_importance(importance: pd.DataFrame, top_n: int = 15) -> Figure:
    """Top random-forest importances, with stationarity features hatched (source Step 11c)."""
    top = importance.head(top_n)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    bars = ax.bar(np.arange(len(top)), top["importance"], color=SERIES[0], edgecolor=SURFACE, linewidth=1)
    for bar, group in zip(bars, top["group"], strict=True):
        if group == "Stationarity":
            bar.set_facecolor(SERIES[1])
            bar.set_hatch("//")
    ax.set_xticks(np.arange(len(top)), top["feature"], rotation=55, ha="right")
    ax.set_ylabel("Mean decrease in impurity")
    ax.set_title(f"Top {top_n} random-forest feature importances (MSM-group vs stationarity features)")
    ax.legend(
        handles=[
            Patch(facecolor=SERIES[0], label="MSM group (includes age, psa)"),
            Patch(facecolor=SERIES[1], hatch="//", label="Stationarity (ADF/KPSS)"),
        ],
        loc="upper right",
    )
    ax.grid(axis="x", visible=False)
    return fig


def _scatter_groups(ax, coords: np.ndarray, groups: np.ndarray, order: Sequence, names: Mapping, colors) -> None:
    for k, group in enumerate(order):
        members = groups == group
        ax.scatter(
            coords[members, 0],
            coords[members, 1],
            s=34,
            color=colors[k % len(colors)],
            marker=MARKERS[k % 3],
            edgecolor=SURFACE,
            linewidth=0.8,
            alpha=0.9,
            label=f"{names.get(group, group)} (n = {members.sum()})",
        )
    ax.legend(loc="best")


def _label_pc_axes(ax, explained_variance: Sequence[float]) -> None:
    ax.set_xlabel(f"PC1 ({100 * explained_variance[0]:.1f}% of variance)")
    ax.set_ylabel(f"PC2 ({100 * explained_variance[1]:.1f}% of variance)")


def plot_pca_tic_vs_msm(
    tic_coords: np.ndarray,
    y_tic: np.ndarray,
    tic_evr: Sequence[float],
    msm_coords: np.ndarray,
    y_msm: np.ndarray,
    msm_evr: Sequence[float],
) -> Figure:
    """PCA of TIC summary statistics next to PCA of the MSM + stationarity features (source Step 13)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    panels = (
        (axes[0], tic_coords, y_tic, tic_evr, "PCA of TIC summary statistics"),
        (axes[1], msm_coords, y_msm, msm_evr, "PCA of MSM + stationarity features"),
    )
    for ax, coords, labels, evr, title in panels:
        _scatter_groups(ax, coords, np.asarray(labels), [0, 1], LABEL_NAMES, SERIES)
        _label_pc_axes(ax, evr)
        ax.set_title(f"{title}\n(colour = cancer label)")
    fig.tight_layout()
    return fig


def plot_pca_before_after(
    before: np.ndarray,
    before_evr: Sequence[float],
    after: np.ndarray,
    after_evr: Sequence[float],
    groups: np.ndarray,
    colour_description: str,
    names: Mapping | None = None,
) -> Figure:
    """PCA of the standardized features before and after ComBat, coloured by ``groups`` (source Steps 14-15).

    With ``names`` (e.g. :data:`LABEL_NAMES`) the groups are the 0/1 labels; without it, the sorted unique values
    (e.g. hospitals).
    """
    groups = np.asarray(groups)
    order = sorted(pd.unique(groups)) if names is None else [0, 1]
    names = names or {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, coords, evr, when in ((axes[0], before, before_evr, "before"), (axes[1], after, after_evr, "after")):
        _scatter_groups(ax, coords, groups, order, names, SERIES)
        _label_pc_axes(ax, evr)
        ax.set_title(f"MSM + stationarity features, PCA {when} ComBat\n(colour = {colour_description})")
    fig.tight_layout()
    return fig


# --- model-comparison figures -------------------------------------------------------------------------------------


def _reference_line(ax, value: float, horizontal: bool = True) -> None:
    label = f"Always predict Cancer: F1 = {value:.3f}"
    draw = ax.axhline if horizontal else ax.axvline
    draw(value, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.3, label=label)


def plot_training_curves(histories: Mapping[str, list[dict]], n_val: int, reference_f1: float | None = None) -> Figure:
    """Validation F1 and loss per epoch for each model, with the always-Cancer reference F1."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharex=True)
    if reference_f1 is not None:
        _reference_line(axes[0], reference_f1)
    for k, (name, history) in enumerate(histories.items()):
        epochs = [h["epoch"] for h in history]
        style = {
            "color": SERIES[k % 3],
            "linestyle": LINESTYLES[k % 3],
            "marker": MARKERS[k % 3],
            "markersize": 4,
            "linewidth": 1.6,
            "label": name,
        }
        axes[0].plot(epochs, [h["val_f1"] for h in history], **style)
        axes[1].plot(epochs, [h["val_loss"] for h in history], **style)
    axes[0].set_ylabel("Validation F1")
    axes[0].set_ylim(0, 1)
    axes[1].set_ylabel("Validation cross-entropy loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend(loc="best")
    axes[0].set_title(f"Validation F1 by epoch (n_val = {n_val})")
    axes[1].set_title("Validation loss by epoch")
    fig.tight_layout()
    return fig


def plot_model_comparison(
    scores: Mapping[str, float], title: str, n_val: int, reference_f1: float | None = None
) -> Figure:
    """Horizontal bars of one F1 score per model, each labelled with its value."""
    names, values = list(scores), list(scores.values())
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(names) + 1.6))
    y = np.arange(len(names))[::-1]
    ax.barh(y, values, color=SERIES[0], height=0.6)
    if reference_f1 is not None:
        _reference_line(ax, reference_f1, horizontal=False)
        ax.legend(loc="lower right")
    for yi, value in zip(y, values, strict=True):
        ax.text(value + 0.01, yi, f"{value:.3f}", va="center", color=INK)
    ax.set_yticks(y, names)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(f"F1 score (validation split, n = {n_val})")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    return fig


def plot_seed_study(table: pd.DataFrame) -> Figure:
    """Best-epoch and final-epoch validation F1 per model across seeds, next to the values in the original report.

    ``table`` is the output of :func:`msm_olfactory.seed_study.run_seed_study`.
    """
    from msm_olfactory.reported import REPORTED_THREE_MODEL_RUNS

    reference = float(table["always_cancer_val_f1"].iloc[0])
    rng = np.random.default_rng(0)  # horizontal jitter only
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    panels = (
        (
            axes[0],
            "best_val_f1",
            "Best validation F1 over all epochs\n(epoch chosen on the validation split, as in the original)",
        ),
        (axes[1], "final_val_f1", "Validation F1 at the final epoch\n(no selection on the validation split)"),
    )
    for ax, metric, title in panels:
        for k, model in enumerate(MODEL_NAMES):
            values = table[f"{model} | {metric}"].to_numpy()
            x = k - 0.12 + rng.uniform(-0.07, 0.07, len(values))
            ax.scatter(
                x,
                values,
                s=34,
                color=SERIES[0],
                marker="o",
                edgecolor=SURFACE,
                linewidth=0.6,
                zorder=3,
                label=f"This reproduction ({len(values)} seed combinations)" if k == 0 else None,
            )
            ax.hlines(
                np.median(values),
                k - 0.24,
                k,
                color=INK,
                linewidth=1.6,
                zorder=4,
                label="Median of reproduction" if k == 0 else None,
            )
            if metric == "best_val_f1":
                reported = [run["f1"][k] for run in REPORTED_THREE_MODEL_RUNS]
                xr = k + 0.14 + rng.uniform(-0.05, 0.05, len(reported))
                ax.scatter(
                    xr,
                    reported,
                    s=46,
                    facecolor="none",
                    edgecolor=SERIES[1],
                    marker="D",
                    linewidth=1.5,
                    zorder=3,
                    label=f"Reported in the original report ({len(reported)} runs)" if k == 0 else None,
                )
        _reference_line(ax, reference)
        ax.set_xticks(range(len(MODEL_NAMES)), ["TIC-image CNN", "MSM-feature MLP", "Multi-modal\ngated ResNet18"])
        ax.set_xlim(-0.6, len(MODEL_NAMES) - 0.4)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="lower left", fontsize=8.5)
    axes[0].set_ylabel("Validation F1 (hold-out split)")
    fig.suptitle("Three-model comparison across random seeds", y=1.02)
    fig.tight_layout()
    return fig


def _log_bars(
    ax,
    labels: Sequence[str],
    values: Sequence[float],
    fmt: Callable[[float], str],
    title: str,
    xlabel: str,
    colors: Sequence[str] | None = None,
) -> None:
    """Horizontal log-scale bars with value labels (log scale: values span several orders of magnitude)."""
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=list(colors) if colors else [SERIES[0]] * len(values), height=0.6)
    for yi, value in zip(y, values, strict=True):
        ax.text(value * 1.15, yi, fmt(value), va="center", color=INK, fontsize=9)
    ax.set_xscale("log")
    ax.set_yticks(y, labels)
    ax.set_xlim(min(values) / 3, max(values) * 30)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)


def _format_parameters(n: float) -> str:
    size = f"{n * 4 / 1e6:.3g} MB" if n > 1e5 else f"{n * 4 / 1e3:.3g} kB"
    return f"{n:,.0f} ({size})"


def _format_seconds(seconds: float) -> str:
    return f"{seconds:.3g} s" if seconds >= 0.1 else f"{1e3 * seconds:.3g} ms"


def plot_swap_c(summary: Mapping) -> Figure:
    """Classifier latency, model size, values per sample and per-stage processing time (log scales).

    ``summary`` is the output of :func:`msm_olfactory.benchmarking.summarize_compute_tradeoffs`.
    """
    measured, calculated = summary["measured"], summary["calculated"]
    models = list(MODEL_NAMES)
    short_names = [SHORT_MODEL_NAMES[m] for m in models]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    _log_bars(
        axes[0, 0],
        short_names,
        [1e3 * measured["forward_seconds_per_batch8_1thread"][m] for m in models],
        lambda v: f"{v:.3g} ms",
        "A. Classifier forward pass (measured)",
        "Median ms per batch of 8, 1 CPU thread (log scale)",
    )
    _log_bars(
        axes[0, 1],
        short_names,
        [calculated["parameters"][m] for m in models],
        _format_parameters,
        "B. Trainable parameters (float32 weight size)",
        "Parameters (log scale)",
    )
    inputs = calculated["classifier_inputs_per_sample"]
    _log_bars(
        axes[1, 0],
        ["Raw TIC (time, intensity)", f"TIC image ({N_IMAGE_ROWS} rows)", "MSM + stationarity vector"],
        [calculated["raw_tic_values_per_sample"], inputs[CNN_NAME], inputs[MLP_NAME]],
        lambda v: f"{v:,.0f} values ({v * 4 / 1e3:.3g} kB)",
        "C. Values per sample at each stage (calculated)",
        "float32 values per sample (log scale)",
        colors=[MUTED, SERIES[0], SERIES[0]],
    )
    extraction = measured["feature_extraction_seconds_per_sample_median"]
    per_sample = measured["forward_seconds_per_sample_1thread"]
    _log_bars(
        axes[1, 1],
        [
            "MSM fit",
            "Rolling ADF + KPSS",
            "TIC image assembly",
            "Intensity-only resample",
            "CNN forward (1 sample)",
            "MLP forward (1 sample)",
        ],
        [
            extraction["msm_fit"],
            extraction["rolling_adf_kpss"],
            extraction["image_assembly"],
            extraction["intensity_only_resample"],
            per_sample[CNN_NAME],
            per_sample[MLP_NAME],
        ],
        _format_seconds,
        "D. Per-sample processing stages (measured)\norange = feature extraction required by BOTH implemented models",
        "Median seconds per sample, single process (log scale)",
        colors=[SERIES[1], SERIES[1], SERIES[0], SERIES[0], SERIES[0], SERIES[0]],
    )
    fig.suptitle("Computational trade study: the MLP is cheap to run, but its inputs are expensive to compute", y=1.01)
    fig.tight_layout()
    return fig
