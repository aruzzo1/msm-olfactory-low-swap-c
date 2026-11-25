"""Configuration: filesystem layout, fixed analytical constants and the experiment settings object.

Three kinds of configuration are kept apart:

* **Filesystem configuration**: where data and outputs live. Every path is relative to the repository root, so the
  project runs from any clone location.
* **Analytical constants**: properties of the method or data format that are not experiment knobs, such as the
  number of TIC-image rows and the p-value thresholds used in the stationarity summaries.
* **Experiment configuration** (:class:`Settings`): values a user may vary. The defaults reproduce the original
  notebook exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --- filesystem configuration -------------------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
REAL_DATA_DIR = DATA_DIR / "raw" / "rotteveel_data"  # extracted Kaggle subset (git-ignored)
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SYNTHETIC_OUTPUT_DIR = OUTPUT_DIR / "synthetic_demo"

TIC_FILENAME = "tic_back.csv"
METADATA_FILENAME = "Samples Demographics.csv"
KAGGLE_DATASET = "adanrott/prostate-cancer-gc-ms-data"

# --- analytical constants -----------------------------------------------------------------------------------------

#: The feature table always has columns for 3 regimes, so a sample that falls back to a 2-regime fit keeps the same
#: columns (padded with NaN).
MAX_REGIMES_IN_FEATURES = 3

#: Significance thresholds used to summarise the rolling ADF/KPSS p-values (as in the source notebook).
P_THRESHOLD = 0.05
P_THRESHOLD_STRICT = 0.01

#: Rows of the TIC image: intensity, MSM regime, 1 - p(ADF), 1 - p(KPSS).
IMAGE_ROW_LABELS = ("intensity", "regime", "ADF (inv)", "KPSS (inv)")
N_IMAGE_ROWS = len(IMAGE_ROW_LABELS)

#: Each image row is scaled between these percentiles so a few extreme peaks do not compress the rest to black.
IMAGE_ROW_PERCENTILES = (1, 99)

#: Display names of the three compared classifiers, in the fixed order used in every table and figure.
CNN_NAME = "TIC-image CNN"
MLP_NAME = "MSM-feature MLP"
MULTIMODAL_NAME = "Multi-modal gated ResNet18"
MODEL_NAMES = (CNN_NAME, MLP_NAME, MULTIMODAL_NAME)

#: Number of Cancer and Non-Cancer samples shown in the TIC-image panel figure.
IMAGE_PANEL_SAMPLES_PER_CLASS = 4

#: PCA random state used for the before/after-ComBat projections (``random_state=42`` in the source). Full-SVD PCA
#: on these small matrices is deterministic, so this does not change the result.
PCA_RANDOM_STATE = 42

#: Bump when the structure of the cached per-sample results changes, so stale caches are recomputed.
DYNAMICS_CACHE_VERSION = 2

# --- experiment configuration -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    """Experiment settings. Every default reproduces the original notebook unless noted otherwise.

    Attributes are grouped as: MSM estimation, rolling stationarity tests, TIC image, train/validation protocol,
    model training, reproducibility and execution.
    """

    # MSM estimation (statsmodels MarkovRegression)
    k_regimes: int = 3  # baseline / intermediate / peak-dominated intensity regimes
    fallback_k_regimes: int = 2  # refit with 2 regimes only if the 3-regime fit raises ValueError
    msm_min_points: int = 200  # shorter series are skipped (real TICs have about 7,335 points)
    em_iter: int = 20  # EM iterations before quasi-Newton optimisation
    search_reps: int = 20  # random perturbed start values; guards against poor local optima
    search_iter: int = 30  # EM iterations per random start

    # Rolling stationarity tests
    station_window: int = 200  # points per window: long enough for ADF lag selection, short enough to be local
    station_step: int = 20  # stride between windows (357 windows for a 7,335-point TIC)
    station_min_points: int = 400  # minimum series length for the rolling tests

    # TIC image
    image_width: int = 256  # time bins after resampling

    # Train / validation protocol: one stratified hold-out split, as in the source
    test_size: float = 0.30
    random_state: int = 42

    # Tabular baselines
    rf_n_estimators: int = 800

    # Deep models
    batch_size: int = 8
    epochs: int = 20
    learning_rate: float = 1e-3

    # Reproducibility (not in the source, which was unseeded). Seeds each sample's MSM start-value search
    # (base_seed + crc32(sample_id)) and PyTorch initialisation and shuffling.
    base_seed: int = 42

    # Sample-ID matching. False = exact match, as in the source (65 of 267 TICs receive labels).
    normalize_sample_ids: bool = False

    # Execution
    workers: int = 1
    run_deep_models: bool = True
    example_sample_id: str | None = None  # None = first sample_id in sorted order, as in the source

    def msm_cache_key(self) -> dict:
        """Settings that determine the per-sample MSM and stationarity results (and nothing else)."""
        return {
            "version": DYNAMICS_CACHE_VERSION,
            "k_regimes": self.k_regimes,
            "fallback_k_regimes": self.fallback_k_regimes,
            "msm_min_points": self.msm_min_points,
            "em_iter": self.em_iter,
            "search_reps": self.search_reps,
            "search_iter": self.search_iter,
            "station_window": self.station_window,
            "station_step": self.station_step,
            "station_min_points": self.station_min_points,
            "base_seed": self.base_seed,
        }


DEFAULT_SETTINGS = Settings()
