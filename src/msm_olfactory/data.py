"""Load GC-MS total ion chromatograms (TICs) and clinical metadata, and join them by sample ID.

Expected layout (the Kaggle dataset ``adanrott/prostate-cancer-gc-ms-data`` of Rotteveel et al., 2025)::

    rotteveel_data/
        Samples Demographics.csv
        <site>/<site>/<SAMPLE>.D/tic_back.csv      e.g. Boston/Boston/BST0001.D/tic_back.csv

Each ``tic_back.csv`` has two header lines (acquisition path and date, then "Start of data points") followed by
``time,intensity`` rows. Clinical metadata is never stored in the TIC files; it is joined on the sample ID taken from
the ``.D`` folder name.
"""

from __future__ import annotations

import logging
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from msm_olfactory.config import METADATA_FILENAME, TIC_FILENAME

logger = logging.getLogger(__name__)

METADATA_RENAME = {
    "Samples": "sample_id",
    "Source": "hospital",
    "prostate.cancer": "cancer_binary",
    "Status": "cancer_status",
    "Class/Gleason Score": "gleason_class",
    "Risk Level": "risk_level",
    "PSA (ng/ml)": "psa",
    "Race": "race",
}
METADATA_TEXT_COLUMNS = ["sample_id", "hospital", "cancer_status", "gleason_class", "risk_level", "race"]
TIC_COLUMNS = ["sample_id", "time", "intensity"]
LABEL_COLUMNS = ["hospital", "cancer_binary", "cancer_status", "gleason_class", "risk_class", "age", "psa", "race"]

#: Fraction of points below which a sample's intensity is taken as its baseline (descriptive column only).
BASELINE_QUANTILE = 0.15


class DataError(RuntimeError):
    """Raised when input data are missing or do not have the expected structure."""


# --- raw files ----------------------------------------------------------------------------------------------------


def extract_required_files(zip_path: Path, dest_dir: Path) -> dict[str, int]:
    """Extract only the TIC files and the demographics table from the Kaggle archive.

    Parameters
    ----------
    zip_path
        Path to ``prostate-cancer-gc-ms-data.zip`` (about 5 GB; 11 GB uncompressed).
    dest_dir
        Destination folder. Existing files are overwritten.

    Returns
    -------
    dict
        Counts of extracted ``tic_files`` and ``metadata_files``.

    Raises
    ------
    DataError
        If the archive is missing or does not contain the expected files.
    """
    zip_path, dest_dir = Path(zip_path), Path(dest_dir)
    if not zip_path.is_file():
        raise DataError(f"Kaggle archive not found: {zip_path}")
    counts = {"tic_files": 0, "metadata_files": 0}
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if name.endswith(TIC_FILENAME):
                counts["tic_files"] += 1
            elif name.rsplit("/", 1)[-1] == METADATA_FILENAME:
                counts["metadata_files"] += 1
            else:
                continue
            archive.extract(name, dest_dir)
    if counts["tic_files"] == 0 or counts["metadata_files"] == 0:
        raise DataError(f"{zip_path} does not look like the Rotteveel GC-MS dataset: {counts}")
    return counts


def sample_id_from_path(tic_path: Path) -> str:
    """Return the sample ID encoded in a TIC path: ``.../BST0001.D/tic_back.csv`` -> ``BST0001``."""
    return Path(tic_path).parent.name.split(".")[0]


def load_tic_file(tic_path: Path) -> pd.DataFrame:
    """Read one ``tic_back.csv`` into columns ``sample_id, time, intensity``.

    Raises
    ------
    DataError
        If the file has no data rows.
    ValueError
        If a time or intensity value is not numeric.
    """
    df = pd.read_csv(tic_path, skiprows=2, sep=r"\s+|,|\t", engine="python", header=None, names=["time", "intensity"])
    if df.empty:
        raise DataError(f"TIC file has no data rows: {tic_path}")
    df["time"] = pd.to_numeric(df["time"], errors="raise")
    df["intensity"] = pd.to_numeric(df["intensity"], errors="raise")
    df["sample_id"] = sample_id_from_path(tic_path)
    return df[TIC_COLUMNS]


def find_tic_files(data_root: Path) -> list[Path]:
    """All ``tic_back.csv`` files under ``data_root``, sorted. Raises :class:`DataError` if there are none."""
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise DataError(f"Data directory not found: {data_root}. See data/README.md.")
    files = sorted(data_root.rglob(TIC_FILENAME))
    if not files:
        raise DataError(f"No {TIC_FILENAME} files under {data_root}. See data/README.md.")
    return files


def load_tic_data(data_root: Path) -> pd.DataFrame:
    """Load every TIC under ``data_root`` into one long table sorted by sample and retention time."""
    frames = [load_tic_file(path) for path in find_tic_files(data_root)]
    return pd.concat(frames, ignore_index=True).sort_values(["sample_id", "time"]).reset_index(drop=True)


def load_metadata(path: Path) -> pd.DataFrame:
    """Load the demographics table with friendly column names and whitespace-stripped text fields.

    Raises
    ------
    DataError
        If the file is missing, empty or lacks any expected column.
    """
    path = Path(path)
    if not path.is_file():
        raise DataError(f"Metadata file not found: {path}")
    meta = pd.read_csv(path).rename(columns=METADATA_RENAME)
    missing = [c for c in METADATA_RENAME.values() if c not in meta.columns]
    if missing:
        raise DataError(f"Metadata file {path.name} is missing columns {missing}")
    if meta.empty:
        raise DataError(f"Metadata file {path.name} has no rows")
    for column in METADATA_TEXT_COLUMNS:
        meta[column] = meta[column].astype(str).str.strip()
    return meta


# --- merge --------------------------------------------------------------------------------------------------------


def _normalized_id(sample_ids: pd.Series) -> pd.Series:
    """Matching key that ignores whitespace and case (``EPPCa 001`` -> ``EPPCA001``)."""
    return sample_ids.astype(str).map(lambda s: re.sub(r"\s+", "", s).upper())


def merge_metadata(tic: pd.DataFrame, meta: pd.DataFrame, normalize_ids: bool = False) -> pd.DataFrame:
    """Left-join clinical metadata onto every TIC row.

    Parameters
    ----------
    tic
        Long TIC table with a ``sample_id`` column.
    meta
        Output of :func:`load_metadata`.
    normalize_ids
        ``False`` (default) reproduces the original exact match on ``sample_id``. With the public dataset this labels
        only Boston and Duke samples (65 of 267). El Paso TIC folders are named like ``EPPCa nnn`` while the metadata
        uses ``EPPCannn``, and Virginia uses ``VGnn`` versus ``Vnn``. ``True`` is an opt-in deviation that ignores
        whitespace when matching (recovering El Paso). It does not map VG to V, because that correspondence is not
        documented.

    Returns
    -------
    pandas.DataFrame
        The TIC rows with metadata columns; unmatched rows have NaN metadata.
    """
    if not normalize_ids:
        return tic.merge(meta, on="sample_id", how="left")
    meta = meta.assign(_key=_normalized_id(meta["sample_id"])).drop(columns=["sample_id"])
    merged = tic.assign(_key=_normalized_id(tic["sample_id"])).merge(meta, on="_key", how="left")
    return merged.drop(columns=["_key"])


def compute_drift_slope(sample_df: pd.DataFrame) -> float:
    """Ordinary least-squares slope of intensity on retention time for one sample (0 if time is constant)."""
    t = sample_df["time"].to_numpy(dtype=float)
    y = sample_df["intensity"].to_numpy(dtype=float)
    denom = np.sum((t - t.mean()) ** 2)
    if denom == 0:
        return 0.0
    return float(np.sum((t - t.mean()) * (y - y.mean())) / denom)


def add_descriptive_columns(tic: pd.DataFrame) -> pd.DataFrame:
    """Add the per-sample descriptive columns created during ingestion in the source notebook.

    ``baseline_mean`` (15th-percentile intensity), ``drift_slope``, ``noise_std_base`` and ``scale_factor`` (always
    1.0) are kept for fidelity. No later step of the analysis uses them.
    """
    tic = tic.copy()
    tic["risk_class"] = tic["risk_level"]
    intensity = tic.groupby("sample_id")["intensity"]
    tic["baseline_mean"] = intensity.transform(lambda x: x.quantile(BASELINE_QUANTILE))
    drift = {sid: compute_drift_slope(group) for sid, group in tic.groupby("sample_id")[["time", "intensity"]]}
    tic["drift_slope"] = tic["sample_id"].map(drift)
    tic["noise_std_base"] = intensity.transform("std")
    tic["scale_factor"] = 1.0
    return tic


def build_tic_table(data_root: Path, normalize_ids: bool = False) -> pd.DataFrame:
    """Load all TICs and the metadata, join them and add descriptive columns (source Steps 4-6)."""
    tic = load_tic_data(data_root)
    meta = load_metadata(Path(data_root) / METADATA_FILENAME)
    return add_descriptive_columns(merge_metadata(tic, meta, normalize_ids))


def load_or_build_tic_table(data_root: Path, cache_path: Path, normalize_ids: bool = False) -> pd.DataFrame:
    """Read the Parquet cache if it exists; otherwise build the table from raw files and cache it."""
    cache_path = Path(cache_path)
    if cache_path.is_file():
        logger.info("Loading cached TIC table (%s)", cache_path.name)
        return pd.read_parquet(cache_path)
    logger.info("Reading raw TIC files from %s", data_root)
    tic = build_tic_table(data_root, normalize_ids)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tic.to_parquet(cache_path, engine="pyarrow", index=False)
    return tic


# --- per-sample views ---------------------------------------------------------------------------------------------


def get_sample_series(tic: pd.DataFrame, sample_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(retention_time, raw_intensity)`` for one sample, sorted by time.

    Raises
    ------
    DataError
        If the sample is not present.
    """
    df = tic.loc[tic["sample_id"] == sample_id, ["time", "intensity"]].sort_values("time")
    if df.empty:
        raise DataError(f"No rows found for sample_id = {sample_id!r}")
    return df["time"].to_numpy(dtype=float), df["intensity"].to_numpy(dtype=float)


def get_sample_labels(tic: pd.DataFrame) -> pd.DataFrame:
    """One row of clinical labels per sample (source Step 9a); ``Age`` is exposed as ``age``."""
    labels = tic.drop_duplicates("sample_id").copy()
    if "Age" in labels.columns and "age" not in labels.columns:
        labels["age"] = labels["Age"]
    columns = ["sample_id"] + [c for c in LABEL_COLUMNS if c in labels.columns]
    return labels[columns].reset_index(drop=True)


def summarize_ingestion(tic: pd.DataFrame) -> dict:
    """Counts describing the merged table: rows, samples, unmatched rows, and labelled samples by site and class."""
    labelled = get_sample_labels(tic).dropna(subset=["cancer_binary"])
    return {
        "tic_rows": len(tic),
        "samples": int(tic["sample_id"].nunique()),
        "rows_missing_metadata": int(tic["cancer_status"].isna().sum()),
        "labelled_samples": len(labelled),
        "labelled_by_hospital": labelled["hospital"].value_counts().sort_index().to_dict(),
        "labelled_by_class": labelled["cancer_binary"].astype(int).value_counts().sort_index().to_dict(),
    }


def cancer_flag(row: pd.Series) -> int | None:
    """Return 1 for Cancer, 0 for Non-Cancer, or None if the row is unlabelled (source Step 12c logic)."""
    if "cancer_binary" in row.index:
        try:
            value = int(row["cancer_binary"])
        except (TypeError, ValueError):  # NaN for samples without metadata
            return None
        return value if value in (0, 1) else None
    status = str(row.get("cancer_status", "")).strip().lower()
    if "non" in status:
        return 0
    if "cancer" in status:
        return 1
    return None


def pick_samples_by_label(tic: pd.DataFrame, n_each: int, random_state: int) -> tuple[list[str], list[str]]:
    """Randomly choose up to ``n_each`` Cancer and Non-Cancer sample IDs (source Step 12c).

    Candidates are taken in ``tic`` order and drawn with ``numpy.random.default_rng(random_state)``, which reproduces
    the source's selection.

    Raises
    ------
    DataError
        If either class has no samples.
    """
    rng = np.random.default_rng(random_state)
    first_rows = tic.drop_duplicates("sample_id").set_index("sample_id")
    flags = {sid: cancer_flag(first_rows.loc[sid]) for sid in tic["sample_id"].unique()}
    cancer = np.array([s for s, f in flags.items() if f == 1])
    noncancer = np.array([s for s, f in flags.items() if f == 0])
    if len(cancer) == 0 or len(noncancer) == 0:
        raise DataError("Need at least one Cancer and one Non-Cancer sample.")
    if len(cancer) > n_each:
        cancer = rng.choice(cancer, size=n_each, replace=False)
    if len(noncancer) > n_each:
        noncancer = rng.choice(noncancer, size=n_each, replace=False)
    return [str(s) for s in cancer], [str(s) for s in noncancer]
