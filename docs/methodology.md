# Methodology

This document describes the analytical pipeline in technical detail: what the original notebook implemented, how
this repository reproduces it, what was verified, and which choices limit the conclusions. Step numbers refer to the
original notebook (`MSM_Data_v18`, exported to HTML), which is the authoritative source for what was implemented.
The accompanying technical report is the source for the motivation and interpretation. Results are in
[`results.md`](results.md), the computational trade study in [`swac_analysis.md`](swac_analysis.md), and the
relationship to the published study in [`project_background.md`](project_background.md).

## 1. Analytical objective

**Proof-of-concept question.** Can Markov-switching and stationarity-derived features preserve comparable
classification performance while substantially reducing computational burden, making machine-olfaction approaches
more viable under SWaP-C constraints? The objective is performance retention at lower cost, not superiority over
the CNN. "Comparable" is defined in [`results.md`](results.md#what-comparable-means-here).

**Why raw TIC intensity is hard to model.** A TIC is the total ion current over about 40 minutes of retention time.
Several properties make it difficult to use directly:

- **Dynamic range.** Intensities span roughly 10^5 to 10^8 counts, and a few large peaks dominate any amplitude-based
  representation.
- **Non-stationarity.** The baseline drifts, and noise level and volatility change along the run as compounds elute,
  so global statistics (mean, variance) mix very different segments.
- **Batch effects.** Instrument, column condition and protocol differ by site. Rotteveel et al. (2025) show that raw
  TIC variation clusters by hospital rather than by disease, and in this data a single principal component of four
  intensity statistics explains 94% of their variance (Step 13).
- **Small labelled samples.** Tens of labelled samples per site make high-capacity models easy to overfit.

**Approach.** Describe each TIC by its *dynamics* rather than its amplitude:

- hidden intensity regimes and their persistence, from a 3-regime MSM;
- where the signal is locally stationary or non-stationary, from rolling ADF and KPSS tests.

These are summarised as 28 per-sample features, harmonized across hospitals with ComBat, and compared in three
classifiers:

1. a CNN on a 4 x 256 "TIC image" (intensity plus MSM and stationarity channels),
2. an MLP on the 30-dimensional per-sample feature vector,
3. a gated fusion of both.

The computational footprint (latency, parameters, input size) is compared as a size, weight, power and cost (SWaP-C)
proxy for edge deployment.

## 2. Inputs

| Input | Description |
|---|---|
| `tic_back.csv` (267 files) | TIC: retention time (min) and total ion intensity, 7,334-7,335 points per sample, from 1.09 to 41.49 min |
| `Samples Demographics.csv` (387 rows) | Sample ID, site, `prostate.cancer` (0/1), status, Gleason class, risk level, age, PSA (ng/ml), race |

**Label coverage.** TICs are joined to metadata by exact sample ID (the `.D` folder name). Only Boston (38) and Duke
(27) IDs match exactly, so **65 of 267 TICs are labelled: 38 Cancer and 27 Non-Cancer**. El Paso folders contain a
space (`EPPCa nnn`) that the metadata (`EPPCannn`) does not. Virginia folders use `VGnn` where the metadata uses
`Vnn`. Every supervised result below therefore comes from **65 samples from two hospitals**. The original notebook's
introduction describes six sites, but its own outputs confirm that only Boston and Duke contributed labels
(`Hospitals present: ['Boston', 'Duke']`). `--normalize-ids` is an opt-in deviation that recovers the El Paso labels
(161 labelled samples). It was not used for any result reported here.

## 3. Preprocessing

For each TIC with raw intensities $x_t$:

$$y_t = \frac{\ell_t - \bar{\ell}}{s_\ell}, \qquad \ell_t = \log(1 + \max(x_t, 0)),$$

which is a log1p compression of the very large peak dynamic range followed by a z-score within the sample. The same
transformed series feeds both the MSM and the stationarity tests. Ingestion also computes per-sample descriptive
columns: the 15th-percentile baseline, an OLS drift slope, the intensity standard deviation, and a constant
`scale_factor = 1`. The original kept these, but **no later step uses them**.

## 4. Markov switching model

Specification (statsmodels `MarkovRegression(y, k_regimes=3, trend="c", switching_variance=False)`):

$$y_t = \mu_{S_t} + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma^2), \quad
S_t \in \{0, 1, 2\}, \quad P(S_t = i \mid S_{t-1} = j) = p_{ij}.$$

- **Regimes differ only in mean level**, with one variance shared by all regimes. In practice the regimes are
  intensity bands: baseline, intermediate, and peak-dominated points (illustrated on a synthetic chromatogram in `outputs/figures/01_*`).
- **Estimation.** Maximum likelihood with 20 EM iterations, then a random search over 20 perturbed starting values
  (`search_reps=20`, 30 EM iterations each), then quasi-Newton optimisation. If the 3-regime fit raises
  `ValueError`, the model is refitted with 2 regimes. On the real data, 266 samples used 3 regimes and one
  (an unlabelled Virginia sample) used 2.
- **Regime inference.** Each point's hard label is the argmax of the smoothed marginal probabilities
  $P(S_t = i \mid y_{1:T})$, computed with the Kim smoother.
- **Transition matrix orientation (verified).** `MarkovRegression.regime_transition_matrix` returns
  $P[i, j] = P(S_t = i \mid S_{t-1} = j)$, so **columns sum to 1**. The feature `trans_i_j` therefore means
  "to regime *i* from regime *j*". The source notebook does not say this; it is checked in `tests/test_msm.py`.
- **Convergence diagnostics.** The source records the log-likelihood, AIC and BIC but performs no explicit
  convergence check. Fit warnings were globally suppressed in the source and are also suppressed here.

**Per-sample MSM features (17):** `n_points`, `k_used`, `llf`, `aic`, `bic`, `regime_frac_{0,1,2}` (share of points
in each regime) and `trans_{i}_{j}` for i, j in {0, 1, 2}. They are padded with NaN when only 2 regimes are used.

**Report vs implementation.** The technical report describes several features that the implemented notebook does
not compute or use:

- regime-specific variances and "volatility measures per regime" (the MSM uses `switching_variance=False`, so all
  regimes share one variance and differ only in mean);
- per-regime means as features;
- drift slope, baseline quantile and noise standard deviation (computed at ingestion but never passed to a model);
- an interquartile spread (never computed).

The report's description of regimes as "low-, medium- and high-volatility" states, and of regime 2 as "peak-like",
is therefore an interpretation rather than a property of the fitted model. Regime numbers are arbitrary (see §9,
item 6). This repository implements only what the notebook implemented.

## 5. Rolling stationarity tests

On 200-point windows taken every 20 points (357 windows per sample), with the window centre time recorded:

- **ADF** (`adfuller`, `autolag="AIC"`). H0 is a unit root, so low p means the window is stationary.
- **KPSS** (`kpss`, `regression="c"`, `nlags="auto"`). H0 is level stationarity, so low p means the window is
  non-stationary. statsmodels interpolates KPSS p-values from a table and **clips them to [0.01, 0.10]**.

**Per-sample stationarity features (11):** `n_station_windows`, the fractions of windows with ADF p < 0.05 or < 0.01
and KPSS p < 0.05 or < 0.01, `frac_both_p_lt_0_05` (both nulls rejected, meaning the two tests disagree),
`frac_adf_high_kpss_low` (ADF p > 0.05 and KPSS p < 0.05, strong evidence of non-stationarity), and the mean and
median of each p-value. Because of the clipping, `frac_kpss_p_lt_0_01` is always 0.

## 6. Modelling table

The merged table (`msm_features_full`, 267 rows) is restricted to the 65 labelled rows. **Every numeric column
except the label becomes a model input: 30 features.** These are the 17 MSM features, the 11 stationarity features,
and **`age` and `psa`**, which are numeric clinical columns that came along with the label merge. The source keeps
them, so this reproduction does too. As a result, the "MSM-only" models are not dynamics-only models.

## 7. Analyses

| Step | Method | Settings |
|---|---|---|
| 11b | Logistic regression (MSM group only; all features) and random forest | One stratified 70/30 split (`random_state=42`), so 45 train and 20 validation. Constant columns dropped (26 features remain). LR uses liblinear, `class_weight="balanced"`, and a scaler fitted on the train split. RF uses 800 trees, balanced, on unscaled features. Metrics: accuracy, F1, ROC-AUC, PR-AUC |
| 13 | PCA | (a) 4 TIC summary statistics (mean, std, max, median intensity); (b) the 30 standardized features |
| 14 | ComBat (`neuroCombat`, parametric empirical Bayes) | batch = hospital, protected covariate = cancer label, fitted on all 65 samples |
| 15 | PCA before and after ComBat | coloured by label |
| 16-18 | TIC-image CNN (3 conv layers, 1,072,066 parameters); MSM-feature MLP (30-64-32-2, 4,130 parameters); gated multi-modal model (1-channel ResNet18 with random initialisation on the image upsampled to 224 x 224, plus a tabular MLP, fused as $g \cdot z_{img} + (1-g) \cdot z_{tab}$ with $g = \sigma(\text{MLP}([z_{img}, z_{tab}]))$; 11,396,931 parameters) | Same 45/20 split. Inputs are the ComBat-harmonized features. Adam (lr = 1e-3), cross-entropy, batch 8, 20 epochs. The reported score is the **best validation F1 over the 20 epochs** |
| 19 | Overhead | Mean forward-pass time per validation batch, trainable parameter counts, inputs per sample |

ComBat model, for feature $g$ of sample $j$ in batch $i$:
$x_{ijg} = \alpha_g + X_j\beta_g + \gamma_{ig} + \delta_{ig}\varepsilon_{ijg}$. The batch location $\gamma$ and scale
$\delta$ are estimated with empirical Bayes shrinkage and removed, while the covariate effect $X_j\beta_g$ (here, the
cancer label) is preserved.

**TIC image (Step 12).** Four rows, each resampled onto 256 evenly spaced retention times:
intensity (linear interpolation), MSM regime (interpolated, rounded, divided by K - 1), $1 - p_{ADF}$ and
$1 - p_{KPSS}$ (interpolated between window centres). Each row is scaled to [0, 1] between its own 1st and 99th
percentiles and stored as 8-bit grayscale.

## 8. Evaluation protocols

### 8.1 Original protocol (reproduced)

A single stratified 70/30 split with `random_state=42` gives 45 training and 20 validation samples (12 Cancer,
8 Non-Cancer). Tabular baselines are scored at a 0.5 threshold. Deep models report the best validation F1 over
20 epochs. No test set is held out.

### 8.2 Reference classifier (added)

The F1 of a constant "always Cancer" prediction on the validation split is $2p/(1+p)$ with $p = 12/20$, which is
**0.750**. It is reported next to every F1 so that F1 values can be read against class prevalence.

### 8.3 Seed-sensitivity study (added; `scripts/run_seed_study.py`)

The original notebook was unseeded, and its report quotes F1 values from six runs. To measure run-to-run variation
under the identical protocol, the pipeline is repeated for:

- 3 MSM start-value seeds (42, 7, 2025), which change the per-sample MSM fits and hence all features;
- 5 PyTorch seeds (42, 1, 2, 3, 4), which change weight initialisation and batch order;

15 combinations in total. The data, split, features, ComBat settings, architectures and training settings are
unchanged. Results go to `outputs/results/seed_study.csv` and figure 11.

### 8.4 Computational measurements (added; `scripts/run_swap_analysis.py`)

- Forward-pass latency: median of 200 repetitions per model and batch size (1 and 8) on one CPU thread.
- Feature-extraction cost per sample: MSM fit, rolling ADF/KPSS and image assembly, timed single-process on five
  labelled samples.
- Parameter counts and input sizes, computed from the models and data shapes.

See [`swac_analysis.md`](swac_analysis.md).

## 9. Findings from the code review

These are the original notebook's choices. They are **reproduced as-is** so the reconstruction stays faithful, and
are listed here so they are not mistaken for endorsements. Items 1-6 affect conclusions.

1. **Label leakage through ComBat.** The scaler and ComBat are fitted on all 65 samples, with the outcome as the
   protected covariate, before the train/validation split. Validation features are therefore adjusted using
   validation labels. *Fix:* fit the scaler and ComBat on the training fold only, then apply the estimates to
   validation; or harmonize without the label as a covariate.
2. **Epoch selection on the validation set.** "Best validation F1 over 20 epochs" selects the model with the same
   20 samples it is scored on, and there is no held-out test set. *Fix:* nested or repeated stratified
   cross-validation with a fixed epoch count or an inner validation split.
3. **Trivial-baseline equivalence.** The headline 0.750 equals the all-Cancer F1 (Section 8.2). *Fix:* report the
   majority/constant baselines, balanced accuracy and ROC-AUC alongside F1.
4. **Clinical covariates inside "MSM" features.** `age` and `psa` are among the 30 inputs to the "MSM-only" MLP and
   the fusion model. PSA is itself a clinical PCa marker. *Fix:* select feature columns explicitly.
5. **"TIC-only" CNN is not TIC-only.** Its image has MSM-regime, ADF and KPSS rows, so the comparison is not
   morphology against dynamics. *Fix:* give the CNN an intensity-only image (one row, or the intensity resampled at
   higher resolution).
6. **Regime label switching.** Regime indices are arbitrary in each fit. For the example sample, the original and this
   reconstruction reach the same optimum (same log-likelihood) but number the regimes differently: the 52.7% regime
   is "0" in the original and "1" here. `regime_frac_r` and `trans_i_j` therefore do not describe the same physical
   regime across samples. *Fix:* relabel regimes by ascending fitted mean $\mu_k$ before extracting features.
7. **Unseeded MSM random starts.** The original's MSM features, and everything downstream, cannot be regenerated
   exactly. This reconstruction seeds each sample's search with `base_seed + crc32(sample_id)`.
8. **Redundant and constant inputs.** `llf`, `aic` and `bic` are perfectly collinear (fixed n and k). `n_points`,
   `k_used`, `n_station_windows` and `frac_kpss_p_lt_0_01` are constant. The tabular step drops constants; the
   ComBat/deep path keeps all 30 columns.
9. **Two sites only.** The ID mismatches (Section 2) remove El Paso and Virginia, which leaves a small, two-batch
   problem.
10. **Minor issues.** (a) The source's label-derivation fallback used `str.contains("cancer")`, which would mark
    "Non-Cancer" as positive. That branch never ran, and the correct rule is used here. (b) The transform's
    `std > 0` guard is defeated by floating-point rounding for constant series; this cannot occur with real TICs.
    (c) The Step 11b markdown mentions temperature and Platt calibration, but no calibration code exists, so none is
    implemented. (d) The descriptive columns from ingestion are never used.

## 10. Deviations from the original implementation

| Change | Reason | Effect on method |
|---|---|---|
| Seeded MSM random-start search (per sample) and seeded PyTorch | Reproducibility | None; same estimator |
| One MSM fit per sample, reused for the features **and** the TIC image (the source refitted the MSM when building images) | Consistency (one set of regimes per sample) and runtime (the source took about 1 h + 15 min + 2 x 20 min) | Images use the same regimes as the features |
| Per-sample fits run in parallel processes | Runtime (about 5 min with 24 workers) | None |
| Only one train/val split, dataset class and training loop (the source defined several near-identical versions) | Consolidation | None; the final version was used |
| "Always Cancer" reference F1 added to results and plots | Context for interpreting F1 | Reporting only |
| Colour palette and figure labels; Step 8, 10 and 12 figures reproduced from the same data | Readability | None |
| Seed-sensitivity study (§8.3) and single-thread latency and feature-extraction timing (§8.4) | Quantify run-to-run variation and end-to-end cost, which the original did not measure | Additional analyses; the original protocol is unchanged |
| OpenAI connectivity check and Kaggle CLI cells removed | Not part of the analysis; they required credentials | None |

## 11. Reproducibility notes

- Python 3.12.14, statsmodels 0.15.0, scikit-learn 1.9.1, numpy 2.5.5, pandas 3.1.1, torch 2.15.0 (CPU),
  neuroCombat 0.2.12. The original Colab versions were not recorded, so `requirements.txt` gives reconstructed lower
  bounds.
- `python scripts/run_analysis.py` is deterministic on a given machine and library set. A repeat run produced
  identical metrics, epochs and feature tables. Different BLAS or library versions may move the MSM optimum
  slightly. The seed study in [`results.md`](results.md#3-seed-sensitivity-study) shows how sensitive the
  downstream scores are to that.
- Intermediate per-sample tables (which contain age, PSA and race) and the TIC images are written to
  `outputs/cache/` and are not committed.

## 12. Limitations

- n = 65 from two hospitals, one 45/20 split, and no external test set.
- The MSM has mean-only switching and fixed K = 3, with no model selection over K and no switching variance.
- Rolling-window tests are not independent (windows overlap by 90%), so fractions of "significant" windows are
  descriptive, not inferential.
- The results say nothing about generalisation to new sites, instruments or populations.

## References

Rotteveel, A., Lee, W.-Y., Kountouri, Z., Stefanou, N., Kivell, H., Gluck, C., Zhang, S., & Mershin, A. (2025).
Towards robust medical machine olfaction: Debiasing GC-MS data enhances prostate cancer diagnosis from urine
volatiles. *PLoS ONE*, 20(5), e0314742. <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12124533/>

Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle.
*Econometrica*, 57(2), 357-384.

Kim, C.-J. (1994). Dynamic linear models with Markov-switching. *Journal of Econometrics*, 60(1-2), 1-22.

Johnson, W. E., Li, C., & Rabinovic, A. (2007). Adjusting batch effects in microarray expression data using empirical
Bayes methods. *Biostatistics*, 8(1), 118-127.

Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root.
*Journal of the American Statistical Association*, 74(366), 427-431.

Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., & Shin, Y. (1992). Testing the null hypothesis of stationarity
against the alternative of a unit root. *Journal of Econometrics*, 54(1-3), 159-178.
