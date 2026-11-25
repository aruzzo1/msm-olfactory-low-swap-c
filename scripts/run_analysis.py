"""Run the full MSM + stationarity analysis pipeline.

Real data (after scripts/prepare_data.py):

    python scripts/run_analysis.py
    python scripts/run_analysis.py --workers 8 --seed 42

Synthetic demo (no download needed; results are meaningless; written to outputs/synthetic_demo/):

    python scripts/run_analysis.py --synthetic
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path

from msm_olfactory.config import DEFAULT_SETTINGS, OUTPUT_DIR, REAL_DATA_DIR, SYNTHETIC_OUTPUT_DIR
from msm_olfactory.data import DataError
from msm_olfactory.pipeline import run_pipeline
from msm_olfactory.synthetic import make_synthetic_dataset
from msm_olfactory.utils import configure_logging

logger = logging.getLogger("run_analysis")


def default_workers() -> int:
    """Leave one CPU free, capped at 8 processes."""
    return max(1, min(8, (os.cpu_count() or 2) - 1))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data",
        "--data-dir",
        dest="data",
        default=str(REAL_DATA_DIR),
        help="extracted dataset folder (default: data/raw/rotteveel_data)",
    )
    parser.add_argument(
        "--output",
        "--output-dir",
        dest="output",
        default=None,
        help="output folder (default: outputs/, or outputs/synthetic_demo/ with --synthetic)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SETTINGS.base_seed,
        help="seed for the MSM start-value search and PyTorch (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers(),
        help="parallel processes for the per-sample MSM fits (default: %(default)s)",
    )
    parser.add_argument("--synthetic", action="store_true", help="generate and analyse SYNTHETIC demo data instead")
    parser.add_argument("--skip-deep", action="store_true", help="skip the CNN, MLP and multimodal models")
    parser.add_argument(
        "--normalize-ids",
        action="store_true",
        help="DEVIATION from the source: match sample IDs ignoring whitespace (labels El Paso samples)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline; returns a process exit code."""
    args = parse_args(argv)
    configure_logging()
    settings = replace(
        DEFAULT_SETTINGS,
        base_seed=args.seed,
        workers=args.workers,
        run_deep_models=not args.skip_deep,
        normalize_sample_ids=args.normalize_ids,
    )
    if args.synthetic:
        output_dir = Path(args.output or SYNTHETIC_OUTPUT_DIR)
        data_dir = make_synthetic_dataset(output_dir / "synthetic_data")
        logger.warning("SYNTHETIC DEMO: randomly generated data with random labels. Scores are meaningless.")
    else:
        output_dir = Path(args.output or OUTPUT_DIR)
        data_dir = Path(args.data)
    try:
        run_pipeline(data_dir, output_dir, settings, is_synthetic=args.synthetic)
    except DataError as exc:
        logger.error("%s", exc)
        logger.error("To try the pipeline without the dataset, run: python scripts/run_analysis.py --synthetic")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
