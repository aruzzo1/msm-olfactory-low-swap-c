"""Values stated in the original technical report, with their provenance. NOT computed by this repository.

Kept in one place so that figures and documents can show them next to reproduced values, always labelled as
"reported". Source documents (not distributed with this repository):

* REPORT     "MSM Power Evaluation on PCa Detection" (technical report, v2, December 2025)
* NB_V12     original notebook version 12 (HTML export)
* NB_V13_15  original notebook versions 13-15 (identical outputs for the cells cited)
* NB_V16_17  original notebook versions 16-17
* NB_V18     original notebook version 18 = the HTML this repository was reconstructed from

All notebook versions use the same MSM, split, model and training settings (checked: K=3, em_iter=20, search_reps=20,
test_size=0.30, random_state=42, 20 epochs, batch 8, lr 1e-3, same three architectures, same ComBat call). Their
results differ because the MSM start-value search and PyTorch training were unseeded.
"""

from __future__ import annotations

from msm_olfactory.config import MODEL_NAMES

# Best validation F1 (selected over 20 epochs on the 20-sample validation split) for the three models.
# Columns: TIC-image CNN, MSM-feature MLP, multi-modal ResNet18.
REPORTED_THREE_MODEL_RUNS = [
    {"run": 1, "f1": (0.75, 0.75, 0.80), "source": "REPORT only (no matching notebook output found)"},
    {"run": 2, "f1": (0.77, 0.78, 0.75), "source": "REPORT only (no matching notebook output found)"},
    {"run": 3, "f1": (0.83, 0.79, 0.79), "source": "REPORT; matches NB_V12 (0.828, 0.786, 0.786)"},
    {"run": 4, "f1": (0.75, 0.75, 0.75), "source": "REPORT; matches NB_V13_15 (0.750, 0.750, 0.750)"},
    {"run": 5, "f1": (0.75, 0.71, 0.75), "source": "REPORT; matches NB_V16_17 / NB_V18 (0.750, 0.714, 0.750)"},
    {"run": 6, "f1": (0.75, 0.71, 0.75), "source": "REPORT; matches NB_V16_17 / NB_V18 (0.750, 0.714, 0.750)"},
]
THREE_MODEL_NAMES = MODEL_NAMES  # column order of the "f1" tuples above

# Step 11b2 logistic regression in NB_V13_15. Features: the 17 "MSM-group" columns (MSM statistics + age + PSA,
# no stationarity features). The report presents it as evidence about "MSM + stationarity" features; in the same
# notebook run, logistic regression on MSM + stationarity features gave F1 0.750 / ROC-AUC 0.604 / PR-AUC 0.713.
REPORTED_LOGREG_11B2 = {
    "f1": 0.786,
    "roc_auc": 0.677,
    "pr_auc": 0.769,
    "confusion_matrix": [[3, 5], [1, 11]],
    "features": "MSM group incl. age and PSA",
    "source": "REPORT p.7-8; NB_V13_15 Step 11b2",
}
REPORTED_LOGREG_11B2_ALL_FEATURES = {
    "f1": 0.750,
    "roc_auc": 0.604,
    "pr_auc": 0.713,
    "source": "NB_V13_15 Step 11b (not quoted in REPORT)",
}

# Forward-pass latency per validation batch (CPU), from Step 19 of each notebook version.
REPORTED_LATENCY_MS = [
    {"cnn_ms": 13.519, "mlp_ms": 0.285, "source": "REPORT p.25 (13.5 ms, 0.29 ms, 47.5x); NB_V13_15"},
    {"cnn_ms": 6.527, "mlp_ms": 0.241, "source": "NB_V16_17 (not quoted in REPORT)"},
    {"cnn_ms": 8.917, "mlp_ms": 0.252, "source": "NB_V18 (not quoted in REPORT)"},
]
REPORTED_PARAMETERS = {"cnn": 1_072_066, "mlp": 4_130, "source": "REPORT p.26; NB_V18"}
REPORTED_INPUTS = {"cnn": 1024, "mlp": 30, "source": "REPORT p.26; NB_V18"}

# Published reference result (Rotteveel et al. 2025): ResNet-18 on TIC images, F1 0.78, recall 0.88.
ROTTEVEEL_2025_F1 = 0.78
