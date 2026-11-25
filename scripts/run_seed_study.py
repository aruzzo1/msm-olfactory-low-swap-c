"""Seed-sensitivity study: rerun the original protocol under several MSM and PyTorch seeds.

Requires the real data (see data/README.md). With the default seeds (3 MSM x 5 PyTorch = 15 runs) this takes about
15-20 minutes with 24 workers. The default seed's MSM fits are reused from scripts/run_analysis.py if cached.

    python scripts/run_seed_study.py --workers 8
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

from msm_olfactory import visualization as viz
from msm_olfactory.config import DEFAULT_SETTINGS, OUTPUT_DIR, REAL_DATA_DIR
from msm_olfactory.data import DataError
from msm_olfactory.seed_study import run_seed_study
from msm_olfactory.utils import configure_logging

logger = logging.getLogger("run_seed_study")

DEFAULT_MSM_SEEDS = [42, 7, 2025]
DEFAULT_TORCH_SEEDS = [42, 1, 2, 3, 4]


def main(argv: list[str] | None = None) -> int:
    """Run the study and write results/seed_study.csv and figure 11; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", "--data-dir", dest="data", default=str(REAL_DATA_DIR))
    parser.add_argument("--output", "--output-dir", dest="output", default=str(OUTPUT_DIR))
    parser.add_argument("--msm-seeds", type=int, nargs="+", default=DEFAULT_MSM_SEEDS)
    parser.add_argument("--torch-seeds", type=int, nargs="+", default=DEFAULT_TORCH_SEEDS)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    configure_logging()
    settings = replace(DEFAULT_SETTINGS, workers=args.workers)
    try:
        table = run_seed_study(Path(args.data), Path(args.output), settings, args.msm_seeds, args.torch_seeds)
    except DataError as exc:
        logger.error("%s", exc)
        return 1
    figure = viz.save_figure(
        viz.plot_seed_study(table), Path(args.output) / "figures" / "11_three_model_f1_seed_study.png"
    )
    logger.info("Figure written to %s", figure)
    return 0


if __name__ == "__main__":
    sys.exit(main())
