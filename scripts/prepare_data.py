"""Extract the files the analysis needs from the Kaggle archive into data/raw/rotteveel_data/.

Download the archive first (free Kaggle account required), either from
https://www.kaggle.com/datasets/adanrott/prostate-cancer-gc-ms-data or with the Kaggle CLI:

    kaggle datasets download -d adanrott/prostate-cancer-gc-ms-data

Then:

    python scripts/prepare_data.py --zip path/to/prostate-cancer-gc-ms-data.zip
"""

from __future__ import annotations

import argparse
import logging
import sys

from msm_olfactory.config import REAL_DATA_DIR
from msm_olfactory.data import DataError, extract_required_files
from msm_olfactory.utils import configure_logging

logger = logging.getLogger("prepare_data")


def main(argv: list[str] | None = None) -> int:
    """Extract the TIC files and demographics table; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", required=True, help="path to prostate-cancer-gc-ms-data.zip")
    parser.add_argument("--dest", default=str(REAL_DATA_DIR), help="destination folder (default: %(default)s)")
    args = parser.parse_args(argv)
    configure_logging()
    try:
        counts = extract_required_files(args.zip, args.dest)
    except DataError as exc:
        logger.error("%s", exc)
        return 1
    logger.info(
        "Extracted %d TIC files and %d metadata file to %s", counts["tic_files"], counts["metadata_files"], args.dest
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
