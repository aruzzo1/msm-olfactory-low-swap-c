# Project background

## The published study this project builds on

Rotteveel et al. (2025) asked whether a machine-learning model could detect prostate cancer from the volatile
organic compounds (VOCs) in urine, in the way trained medical detection dogs do. Their approach does not identify
individual compounds. It treats each GC-MS chromatogram as a "scent character" signal.

As reported in their paper:

- **Data.** 387 urine samples from four US medical centres; 365 were used after excluding corrupted data
  (125 biopsy-negative controls, the rest PCa of varying risk).
- **Debiasing.** Empirical Bayes batch correction, baseline-drift removal, and domain-adversarial learning
  against the sample's source site.
- **Model.** A pretrained ResNet18 on image representations of the GC-MS data, with a cancer-risk head and an
  adversarial source head.
- **Results.** Under five-fold cross-validation: accuracy 75%, recall 88% and F1 0.78 for PCa-positive versus
  negative.

The authors published the GC-MS data on Kaggle (`adanrott/prostate-cancer-gc-ms-data`). That public release is the
only data this project uses.

**This repository is not part of that publication.** The study, the sample collection and the dataset are the work
of Rotteveel et al. This repository's author is not an author of the paper or the dataset, and nothing here should be
read as a re-analysis endorsed by them.
The comparisons with their results are limited to what their paper reports.

## Application context

The longer-term interest behind this work is **machine olfaction under tight resource limits**: chemical and VOC
sensing on small devices that must process data at the edge. Such devices often operate in disconnected or
bandwidth-constrained environments, where streaming raw data to a server is not an option. That includes field,
remote and defence-related sensing, where size, weight, power and cost (SWaP-C) are first-order design constraints.
In that setting, classification accuracy has to be weighed together with model size, memory, input bandwidth and
compute per inference.

The prostate-cancer GC-MS dataset served as a well-documented, publicly available **test bed** for that question. The
aim was not to build a diagnostic tool.

## What this project explored

The published work learns from the *shape* of a chromatogram: peak positions, heights and widths. This project tested
whether a representation of the signal's *dynamics* could be used more cheaply. The proof-of-concept question was:

> Can Markov-switching and stationarity-derived features preserve comparable classification performance while
> substantially reducing computational burden, making machine-olfaction approaches more viable under SWaP-C
> constraints?

The goal was **not** to show that this approach outperforms a CNN. It was to establish whether a lower-dimensional
representation could cut model complexity and inference cost without materially sacrificing performance.

Specifically, it added:

| Component | Origin |
|---|---|
| GC-MS TICs, clinical labels, site structure | Rotteveel et al. public dataset |
| Batch correction with empirical Bayes (ComBat) | Same idea as Rotteveel et al., applied here to engineered features with `neuroCombat` |
| CNN on an image representation of the TIC | Inspired by Rotteveel et al.; implemented here as a small 3-layer CNN on a 4 x 256 multichannel image (not their architecture, image format or debiasing pipeline) |
| 3-regime Markov switching model per chromatogram | New in this project |
| Rolling ADF / KPSS stationarity profiles | New in this project |
| 30-feature dynamic "fingerprint" and an MLP on it | New in this project |
| Gated image + feature fusion model | New in this project |
| Classifier latency, model size, input size (SWaP-C) | New in this project |

Because the CNN, image format, debiasing, sample set (65 labelled samples from two sites; see `data/README.md`) and
evaluation protocol (one 45/20 split) all differ, **results here are not directly comparable to the published
F1 of 0.78**.

## History of this implementation

| When | Artefact | Notes |
|---|---|---|
| Dec 2025 | Google Colab notebook, versions 12-18 | Same pipeline throughout. The random seeds were not fixed, so each run gave different scores |
| Dec 2025 | Technical report, "MSM Power Evaluation on PCa Detection" (v2) | Narrative report with results quoted from several notebook runs and versions. Not distributed here |
| Nov 2026 | This repository | Reconstructed from notebook v18 as a tested, seeded, reproducible package; results re-derived from the public data; claims re-checked |

[`results.md`](results.md) separates what was reproduced from what is only stated in the report.

## References

Rotteveel, A., Lee, W.-Y., Kountouri, Z., Stefanou, N., Kivell, H., Gluck, C., Zhang, S., & Mershin, A. (2025).
Towards robust medical machine olfaction: Debiasing GC-MS data enhances prostate cancer diagnosis from urine
volatiles. *PLoS ONE*, 20(5), e0314742. Open access: <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12124533/>.
Preprint: <https://doi.org/10.1101/2024.11.19.624273>.
