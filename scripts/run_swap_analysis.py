"""Computational (SWaP-C) trade study: classifier latency, model size, input size and feature-extraction cost.

Uses the cached TIC table from scripts/run_analysis.py (real data, or --synthetic for the demo). Writes
outputs/results/swap_c_analysis.json and outputs/figures/12_swap_c_trade_study.png.

    python scripts/run_swap_analysis.py
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from msm_olfactory import benchmarking
from msm_olfactory import visualization as viz
from msm_olfactory.config import CNN_NAME, DEFAULT_SETTINGS, MLP_NAME, OUTPUT_DIR, SYNTHETIC_OUTPUT_DIR
from msm_olfactory.data import get_sample_labels
from msm_olfactory.utils import configure_logging, write_json

logger = logging.getLogger("run_swap_analysis")

#: Width of the MSM-feature vector given to the MLP (17 MSM + 11 stationarity + age + PSA, as in the source).
FEATURE_VECTOR_SIZE = 30


def main(argv: list[str] | None = None) -> int:
    """Run the benchmarks and write the JSON summary and figure 12; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--synthetic", action="store_true", help="use the synthetic demo outputs")
    parser.add_argument("--n-samples", type=int, default=5, help="labelled samples timed for feature extraction")
    args = parser.parse_args(argv)
    configure_logging()

    out_dir = SYNTHETIC_OUTPUT_DIR if args.synthetic else OUTPUT_DIR
    cache = out_dir / "cache" / "tic_all_with_metadata.parquet"
    if not cache.is_file():
        logger.error("%s not found. Run scripts/run_analysis.py first.", cache)
        return 1
    settings = DEFAULT_SETTINGS
    tic = pd.read_parquet(cache)
    labelled = get_sample_labels(tic).dropna(subset=["cancer_binary"])["sample_id"].tolist()
    timed = labelled[: args.n_samples]
    raw_points = int(tic.groupby("sample_id").size().median())

    logger.info("Timing classifiers (1 CPU thread) and feature extraction on %d samples...", len(timed))
    classifier_rows = benchmarking.benchmark_classifier_inference(FEATURE_VECTOR_SIZE, settings.image_width)
    extraction = benchmarking.benchmark_feature_extraction(tic, timed, settings)
    summary = benchmarking.summarize_compute_tradeoffs(
        classifier_rows, extraction, FEATURE_VECTOR_SIZE, settings.image_width, raw_points
    )
    summary["classifier_timings"] = classifier_rows
    # Per-sample timings are published without sample IDs.
    summary["feature_extraction_timings"] = extraction.drop(columns=["sample_id"]).to_dict(orient="records")
    summary["data_is_synthetic"] = bool(args.synthetic)

    results = write_json(out_dir / "results" / "swap_c_analysis.json", summary)
    viz.save_figure(viz.plot_swap_c(summary), out_dir / "figures" / "12_swap_c_trade_study.png")
    measured, calculated = summary["measured"], summary["calculated"]
    logger.info(
        "Forward pass, batch of 8: CNN %.3f ms vs MLP %.4f ms (%.1fx)",
        1e3 * measured["forward_seconds_per_batch8_1thread"][CNN_NAME],
        1e3 * measured["forward_seconds_per_batch8_1thread"][MLP_NAME],
        calculated["speedup_cnn_over_mlp_batch8"],
    )
    logger.info("Feature extraction per sample (s): %s", measured["feature_extraction_seconds_per_sample_median"])
    logger.info("Wrote %s", results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
