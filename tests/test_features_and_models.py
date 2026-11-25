"""Feature assembly, the shared evaluation split, ComBat, tabular baselines and the neural models."""

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.model_selection import train_test_split

from msm_olfactory import batch_correction, deep_models, features, images, tabular_models
from msm_olfactory.config import MODEL_NAMES, Settings
from msm_olfactory.evaluation import always_cancer_f1, evaluate_probabilities, stratified_holdout_indices


def _feature_table(n=24, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "sample_id": [f"S{i}" for i in range(n)],
            "n_points": 7335,
            "llf": rng.normal(size=n),
            "trans_0_0": rng.random(n),
            "hospital": ["Boston"] * (n // 2) + ["Duke"] * (n - n // 2),
            "cancer_binary": [0.0, 1.0] * (n // 2),
            "age": rng.integers(50, 80, n),
            "psa": rng.random(n) * 10,
            "frac_adf_p_lt_0_05": rng.random(n),
            "mean_kpss_p": rng.random(n),
        }
    )
    df.loc[0, "cancer_binary"] = np.nan  # an unlabelled sample
    return df


# --- features -----------------------------------------------------------------------------------------------------


def test_model_matrix_matches_source_selection():
    matrix = features.build_model_matrix(_feature_table())
    assert len(matrix.y) == 23 and set(matrix.y) == {0, 1}
    assert matrix.X.shape == (23, 7) and len(matrix.sample_ids) == len(matrix.batch) == 23
    assert "cancer_binary" not in matrix.numeric.columns
    assert {"age", "psa", "n_points"} <= set(matrix.numeric.columns)  # kept exactly as in the source


def test_model_matrix_validates_input():
    with pytest.raises(KeyError, match="hospital"):
        features.build_model_matrix(_feature_table().drop(columns=["hospital"]))
    one_class = _feature_table().assign(cancer_binary=1.0)
    with pytest.raises(ValueError, match="both classes"):
        features.build_model_matrix(one_class)


def test_tabular_features_drop_constants_and_group_columns():
    tab = features.prepare_tabular_features(_feature_table())
    assert "n_points" in tab.dropped_constant and "n_points" not in tab.X.columns
    assert set(tab.stationarity_features) == {"frac_adf_p_lt_0_05", "mean_kpss_p"}
    assert {"age", "psa", "llf"} <= set(tab.msm_features)
    with pytest.raises(KeyError):
        features.prepare_tabular_features(pd.DataFrame({"x": [1.0]}))


# --- evaluation ---------------------------------------------------------------------------------------------------


def test_holdout_split_is_stratified_deterministic_and_matches_source_call():
    labels = np.array([1] * 38 + [0] * 27)  # the real labelled class balance
    train_idx, val_idx = stratified_holdout_indices(labels, Settings())
    assert len(val_idx) == 20 and not set(train_idx) & set(val_idx)
    assert labels[val_idx].sum() == 12  # 12 Cancer, 8 Non-Cancer, as in the source
    # identical to the source's call on a list of records (which the deep-model stage used)
    records = [f"r{i}" for i in range(len(labels))]
    _, val_records = train_test_split(records, test_size=0.30, random_state=42, stratify=labels)
    assert [records[i] for i in val_idx] == val_records


def test_holdout_split_rejects_too_few_samples():
    with pytest.raises(ValueError, match="at least 2 samples per class"):
        stratified_holdout_indices([1, 1, 1, 0], Settings())


def test_metrics_and_reference():
    assert always_cancer_f1([1] * 12 + [0] * 8) == pytest.approx(0.75)
    m = evaluate_probabilities(np.array([0, 0, 1, 1]), np.array([0.1, 0.6, 0.7, 0.9]))
    assert m["accuracy"] == 0.75 and m["confusion_matrix"] == [[1, 1], [0, 2]] and m["roc_auc"] == 1.0


# --- batch correction and baselines -------------------------------------------------------------------------------


def test_combat_removes_additive_batch_shift():
    rng = np.random.default_rng(0)
    n, p = 40, 5
    batch = np.array(["A"] * 20 + ["B"] * 20)
    X = rng.normal(size=(n, p)) + np.where(batch == "B", 3.0, 0.0)[:, None]
    X_std = batch_correction.standardize_features(X)
    Xc = batch_correction.apply_combat_correction(
        X_std, batch, np.tile([0, 1], 20), np.array([f"s{i}" for i in range(n)])
    )

    def gap(M):
        return np.abs(M[batch == "A"].mean(0) - M[batch == "B"].mean(0)).mean()

    assert Xc.shape == (n, p) and gap(Xc) < 0.2 * gap(X_std)


def test_combat_validates_batches():
    with pytest.raises(ValueError, match="two batches"):
        batch_correction.apply_combat_correction(np.zeros((4, 2)), np.array(["A"] * 4), np.zeros(4), np.arange(4))


def test_tabular_baselines_report_metrics():
    result = tabular_models.run_tabular_baselines(features.prepare_tabular_features(_feature_table(40)), Settings())
    for key in ("logreg_msm_group", "logreg_all_features", "random_forest_all_features"):
        assert 0.0 <= result.metrics[key]["f1"] <= 1.0
    assert result.importance["importance"].sum() == pytest.approx(1.0)
    assert set(result.importance["group"]) <= {"MSM", "Stationarity"}


# --- neural models ------------------------------------------------------------------------------------------------


def test_parameter_counts_match_the_original_notebook():
    # values printed in the source notebook (Step 19)
    assert deep_models.count_parameters(deep_models.TicOnlyCNN()) == 1_072_066
    assert deep_models.count_parameters(deep_models.MsmOnlyMLP(30)) == 4_130


def test_model_output_shapes_and_factory():
    img, tab = torch.rand(4, 1, 4, 256), torch.rand(4, 30)
    models = deep_models.build_models(30, 256)
    assert tuple(models) == MODEL_NAMES
    for model in models.values():
        model.eval()
        assert model(img, tab).shape == (4, 2)


def test_cnn_respects_configured_image_width():
    # Regression test: the CNN must be built for the configured image width (was hard-coded to 256).
    cnn = deep_models.build_models(30, 128)[MODEL_NAMES[0]]
    assert cnn(torch.rand(2, 1, 4, 128), None).shape == (2, 2)


def test_dataset_returns_aligned_tensors(tmp_path):
    path = images.save_tic_image(np.full((4, 256), 128, np.uint8), tmp_path / "a.png")
    ds = deep_models.TicMsmDataset(
        [{"sample_id": "a", "path": str(path), "row": 1, "label": 1}], np.arange(60, dtype=float).reshape(2, 30)
    )
    img, tab, label = ds[0]
    assert img.shape == (1, 4, 256) and tab[0].item() == 30.0 and label.item() == 1


def test_training_rejects_misaligned_features(tmp_path):
    records = [{"sample_id": "a", "path": "x.png", "row": 0, "label": 1}]
    with pytest.raises(ValueError, match="rows for"):
        deep_models.train_and_compare_models(records, [0], [0], np.zeros((2, 30)), Settings(), 0)
    with pytest.raises(ValueError, match="NaN"):
        deep_models.train_and_compare_models(records, [0], [0], np.full((1, 30), np.nan), Settings(), 0)
