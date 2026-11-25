# Code architecture

The analysis is a Python package (`src/msm_olfactory/`) driven by thin command-line scripts (`scripts/`). The
notebook (`notebooks/MSM_analysis.ipynb`) explains the method and calls the same package functions; it is not a
second implementation.

## Data flow

```
 Kaggle archive ──► scripts/prepare_data.py ──► data/raw/rotteveel_data/   (TIC CSVs + demographics, git-ignored)
                                                        │
 scripts/run_analysis.py ──► pipeline.run_pipeline      ▼
                               │   data ............ load TICs + metadata, sample-ID join, schema checks
                               │   sample_dynamics . per sample, in parallel, seeded, cached:
                               │      ├─ preprocessing .. log1p + z-score
                               │      ├─ msm ............ 3-regime Markov switching fit → 17 MSM features
                               │      └─ stationarity ... rolling ADF / KPSS → 11 stationarity features
                               │   features ........ feature tables, modelling matrix, tabular inputs
                               │   tabular_models .. logistic regression, random forest    ─┐
                               │   batch_correction  PCA, ComBat by hospital                │  evaluation:
                               │   images .......... 4 × 256 TIC image per sample           │  shared split,
                               │   deep_models ..... TIC-image CNN, feature MLP,            │  metrics,
                               │                     gated multimodal ResNet18 ─────────────┘  reference F1
                               │   visualization ... public figures (sample-level panel kept local)
                               ▼
               outputs/figures/   outputs/results/run_summary.json   outputs/cache/ (git-ignored)

 scripts/run_seed_study.py ──► seed_study.run_seed_study (reuses the pipeline stages) ──► seed_study.csv, figure 11
 scripts/run_swap_analysis.py ► benchmarking (latency, model size, feature-extraction cost) ► swap_c_analysis.json, fig 12
```

## Modules

| Module | Responsibility |
|---|---|
| `config.py` | Filesystem paths, analytical constants (image rows, p-value thresholds, model names) and the `Settings` experiment configuration. Defaults reproduce the source notebook |
| `data.py` | Extract the Kaggle subset; load and validate TIC files and metadata; sample-ID join; per-sample views; ingestion summary |
| `preprocessing.py` | `log_zscore_transform`, shared by the MSM and the stationarity tests |
| `msm.py` | `fit_markov_switching_model` (3 regimes, 2-regime fallback) and `extract_msm_features` |
| `stationarity.py` | `rolling_stationarity_tests` and `compute_stationarity_features` |
| `sample_dynamics.py` | Runs both per-sample analyses in parallel processes with per-sample seeds and an on-disk cache; one fit feeds both features and images |
| `features.py` | The three feature tables, the 30-column `ModelMatrix`, the tabular-baseline inputs, TIC summary statistics |
| `images.py` | The 4-row TIC image (intensity, regime, 1-p(ADF), 1-p(KPSS)) |
| `batch_correction.py` | Standardisation, `apply_combat_correction` (batch = hospital, label protected), PCA projections |
| `evaluation.py` | The single stratified hold-out split, classification metrics, the always-Cancer reference F1 |
| `tabular_models.py` | Logistic regression (MSM group; all features) and random forest; coefficients and importances |
| `deep_models.py` | Dataset, the three PyTorch models, training loop, `train_and_compare_models`, Step-19 timing |
| `benchmarking.py` | SWaP-C trade study: single-thread latency, feature-extraction timing, derived ratios |
| `visualization.py` | All figures; functions return matplotlib figures and never modify data |
| `pipeline.py` | Orchestration: one small function per source stage, `run_pipeline` entry point |
| `seed_study.py` | Repeats the protocol under several MSM and PyTorch seeds |
| `reported.py` | Values quoted in the original report, with provenance, kept separate from computed results |
| `synthetic.py` | Clearly labelled random demo data in the real file layout |
| `utils.py` | Logging setup, JSON helpers, silencing of known third-party console output |

## Where things happen

| Concern | Location |
|---|---|
| Model **training** | `tabular_models.run_tabular_baselines` (scikit-learn) and `deep_models.train_model` / `train_and_compare_models` (PyTorch) |
| **Evaluation** | `evaluation.py` defines the split and the metrics, which both model modules use. Results are collected in `pipeline.run_tabular_stage` and `pipeline.run_deep_model_stage` |
| **Visualization** | `visualization.py`, called from the pipeline stages, the seed study and the trade-study script |
| **Reproducibility** | Per-sample MSM seeds (`sample_dynamics.sample_seed`), PyTorch and DataLoader seeds (`deep_models.train_and_compare_models`), fixed split (`evaluation.stratified_holdout_indices`) |
| **Caching** | `outputs/cache/`: TIC table (Parquet), per-sample dynamics (pickle keyed by `Settings.msm_cache_key`), TIC images |
| **Configuration** | `config.Settings` (experiment), module-level constants in `config.py` (analytical), paths in `config.py` (filesystem) |

## Design choices

- **One implementation.** The scripts, the notebook and the tests all call the package. Nothing analytical is
  duplicated in notebook cells.
- **Pure stages, explicit inputs.** Each pipeline stage takes data and settings and returns results. File output is
  limited to figures, aggregate results and the cache.
- **Lazy heavy imports.** `import msm_olfactory` does not load PyTorch, so the parallel MSM worker processes stay
  light. Neural-network names are resolved on first use.
- **Fail clearly.** Missing or malformed input raises `DataError`, and analysis preconditions raise `ValueError` or
  `RuntimeError` with an explanatory message. Examples: a single class, too few samples per class, misaligned
  feature matrices, or every MSM fit failing.
- **Provenance.** Values quoted from the original report live only in `reported.py`, and are drawn in figures as
  "reported", never as reproduced.
