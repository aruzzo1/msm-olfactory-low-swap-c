# Data

## Source data is not redistributed

**This repository contains no source data.** The original model-input data is **intentionally excluded**:

- GC-MS total ion chromatograms
- clinical metadata
- sample-level labels
- any per-sample table derived from them

The repository contains only code, methodology, tests, reproducibility tooling, aggregate results, aggregate
figures, and **synthetic** illustrative figures and test fixtures. Sample-level chromatogram visualizations are intentionally excluded from the public repository. Public figures are limited to synthetic illustrative examples and aggregate analytical results. **No proprietary RealNose or other partner data
is included.**

The research dataset is not the author's. It belongs to its creators: the GC-MS prostate-cancer data accompanying
Rotteveel et al. (2025). The paper's data-availability statement reads: *"All GC-MS files are available from the
Kaggle website at the url: https://www.kaggle.com/datasets/adanrott/prostate-cancer-gc-ms-data"*. To reproduce the
analysis, **obtain the data independently from that original source** and follow its terms of use. This
repository makes no claim about the dataset's redistribution licence; the CC BY licence of the published article
covers the article, not necessarily the dataset.

- Kaggle dataset `adanrott/prostate-cancer-gc-ms-data`: <https://www.kaggle.com/datasets/adanrott/prostate-cancer-gc-ms-data>
- About 5.1 GB zipped (11 GB uncompressed); the analysis reads about 35 MB of it.

Synthetic, randomly generated fixtures (`src/msm_olfactory/synthetic.py`) exist **only for testing** that the
software runs. They are not real chromatograms or patient records, and any score computed on them is meaningless.

## Obtaining and placing the data

1. Download the archive from the source above (a free Kaggle account is required), in a browser or with the
   Kaggle CLI:

   ```powershell
   kaggle datasets download -d adanrott/prostate-cancer-gc-ms-data
   ```

2. Extract only the files the analysis needs into the git-ignored `data/raw/` folder:

   ```powershell
   python scripts\prepare_data.py --zip path\to\prostate-cancer-gc-ms-data.zip
   ```

   ```
   data/raw/rotteveel_data/                 (git-ignored; never committed)
       Samples Demographics.csv             clinical metadata (387 rows)
       <site>/<site>/<ID>.D/tic_back.csv    one TIC per sample (267 files in four site folders)
   ```

Without the data, `python scripts/run_analysis.py` stops with a clear message, and
`python scripts/run_analysis.py --synthetic` runs the pipeline on synthetic fixtures.

## Expected file formats

**`tic_back.csv`**: two header lines (acquisition path and date, then `Start of data points`), followed by
`retention_time_min,intensity` rows (about 7,335 rows per sample).

**`Samples Demographics.csv`**: columns `S/N, Source, Samples, prostate.cancer, Status, Class/Gleason Score,
Risk Level, Age, PSA (ng/ml), Race`. `Samples` is the sample ID and `prostate.cancer` is the 0/1 label.

## Sample-ID matching

TICs are joined to the metadata on the sample ID taken from the `.D` folder name. An exact match, as in the original
notebook, labels **only 65 of 267 TICs (Boston 38, Duke 27)**, because ID formats differ between the TIC folders and
the metadata for two sites:

| Site | TIC folder ID format | Metadata ID format | Exact match |
|---|---|---|---|
| Boston, Duke | `BSTnnnn`, `DKnnnn` | `BSTnnnn`, `DKnnnn` | yes (65 samples) |
| El Paso | `EPPCa nnn` (with a space) | `EPPCannn` | no |
| Virginia 1 | `VGnn` | `Vnn` | no |

`scripts/run_analysis.py --normalize-ids` is an **opt-in deviation** from the original. It ignores whitespace when
matching, which labels 161 samples. It does not map `VG` to `V`, because that correspondence is not documented.

## What is and is not published

| Content | Location | Published? |
|---|---|---|
| Source TICs and clinical metadata | `data/raw/` | **No**: excluded by `.gitignore` |
| Per-sample derived tables (features, labels, age, PSA, race), fitted dynamics, TIC images | `outputs/cache/` | **No**: excluded |
| Synthetic test fixtures | generated at run time in `outputs/synthetic_demo/` | **No**: generated, not stored |
| Aggregate results (metrics, importances, explained variance, timings) | `outputs/results/` | Yes |
| Figures: synthetic illustrative examples (01, 02, 04) and aggregate results (03, 06-12) | `outputs/figures/` | Yes |
| Sample-level chromatogram figures (local inspection only) | `outputs/cache/sample_level_figures/` | **No**: excluded |

`.gitignore` also blocks sample-level export formats anywhere in the tree (Parquet, pickle, NumPy, HDF5, spreadsheets,
archives, and CSV/JSON outside `outputs/results/`).

## Reference

Rotteveel, A., Lee, W.-Y., Kountouri, Z., Stefanou, N., Kivell, H., Gluck, C., Zhang, S., & Mershin, A. (2025).
Towards robust medical machine olfaction: Debiasing GC-MS data enhances prostate cancer diagnosis from urine
volatiles. *PLoS ONE*, 20(5), e0314742. <https://doi.org/10.1371/journal.pone.0314742>
