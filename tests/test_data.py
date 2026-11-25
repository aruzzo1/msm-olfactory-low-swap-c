"""Data loading, schema validation and the TIC-metadata join."""

import zipfile

import numpy as np
import pandas as pd
import pytest

from msm_olfactory import data


def _write_tic(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"{t},{v}" for t, v in rows)
    path.write_text(f"D:\\acq\\X.D Tue Apr 26 2022\nStart of data points\n{body}\n")


def _write_meta(path, samples):
    n = len(samples)
    pd.DataFrame(
        {
            "S/N": range(1, n + 1),
            "Source": ["Boston"] * n,
            "Samples": samples,
            "prostate.cancer": [1, 0][:n],
            "Status": ["Cancer", "Non-Cancer"][:n],
            "Class/Gleason Score": ["6", "BPH"][:n],
            "Risk Level": ["Low-Risk", "Control"][:n],
            "Age": [66, 63][:n],
            "PSA (ng/ml)": [4.1, 5.8][:n],
            "Race": [" Caucasian ", "Others"][:n],
        }
    ).to_csv(path, index=False)


def test_load_tic_file_skips_header_and_takes_id_from_folder(tmp_path):
    path = tmp_path / "EPP" / "EPP" / "EPPCa 001.D" / "tic_back.csv"
    _write_tic(path, [(1.09, 500), (1.10, 510)])
    df = data.load_tic_file(path)
    assert list(df.columns) == ["sample_id", "time", "intensity"]
    assert df["sample_id"].unique().tolist() == ["EPPCa 001"]
    assert df["intensity"].tolist() == [500, 510]


def test_empty_tic_file_is_rejected(tmp_path):
    path = tmp_path / "A" / "A" / "S1.D" / "tic_back.csv"
    path.parent.mkdir(parents=True)
    path.write_text("header\nStart of data points\n")
    with pytest.raises(data.DataError, match="no data rows"):
        data.load_tic_file(path)


def test_load_metadata_renames_and_strips(tmp_path):
    _write_meta(tmp_path / "meta.csv", ["BST1", "BST2"])
    meta = data.load_metadata(tmp_path / "meta.csv")
    assert {"sample_id", "hospital", "cancer_binary", "psa"} <= set(meta.columns)
    assert meta["race"].tolist() == ["Caucasian", "Others"]


def test_load_metadata_rejects_wrong_schema_and_missing_file(tmp_path):
    pd.DataFrame({"a": [1]}).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(data.DataError, match="missing columns"):
        data.load_metadata(tmp_path / "bad.csv")
    with pytest.raises(data.DataError, match="not found"):
        data.load_metadata(tmp_path / "absent.csv")


def test_missing_or_empty_data_dir_raises(tmp_path):
    with pytest.raises(data.DataError, match="not found"):
        data.find_tic_files(tmp_path / "nope")
    with pytest.raises(data.DataError, match=r"No tic_back\.csv"):
        data.find_tic_files(tmp_path)


def test_exact_merge_reproduces_source_and_normalized_merge_is_opt_in(tmp_path):
    tic = pd.DataFrame({"sample_id": ["EPPCa 001", "BST1"], "time": [1.0, 1.0], "intensity": [1.0, 2.0]})
    _write_meta(tmp_path / "m.csv", ["EPPCa001", "BST1"])
    meta = data.load_metadata(tmp_path / "m.csv")
    exact = data.merge_metadata(tic, meta)
    assert exact.loc[exact.sample_id == "EPPCa 001", "cancer_binary"].isna().all()  # source behaviour
    normalized = data.merge_metadata(tic, meta, normalize_ids=True)
    assert normalized["cancer_binary"].notna().all()
    assert normalized["sample_id"].tolist() == ["EPPCa 001", "BST1"]  # TIC ids are kept


def test_drift_slope_is_ols_slope():
    df = pd.DataFrame({"time": [0.0, 1.0, 2.0, 3.0], "intensity": [1.0, 3.0, 5.0, 7.0]})
    assert data.compute_drift_slope(df) == pytest.approx(2.0)
    assert data.compute_drift_slope(pd.DataFrame({"time": [1.0, 1.0], "intensity": [1.0, 2.0]})) == 0.0


def test_build_tic_table_schema_and_summary(synthetic_root):
    tic = data.build_tic_table(synthetic_root)
    for column in ["sample_id", "time", "intensity", "hospital", "cancer_binary", "baseline_mean", "drift_slope"]:
        assert column in tic.columns
    assert tic["sample_id"].nunique() == 12
    assert tic.groupby("sample_id")["time"].apply(lambda t: t.is_monotonic_increasing).all()
    summary = data.summarize_ingestion(tic)
    assert summary["samples"] == 12 and summary["labelled_samples"] == 12 and summary["rows_missing_metadata"] == 0


def test_extract_required_files_takes_only_needed_members(tmp_path):
    zpath = tmp_path / "k.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("Boston/Boston/BST1.D/tic_back.csv", "h\nh\n1,2\n")
        z.writestr("Boston/Boston/BST1.D/other.xml", "<x/>")
        z.writestr("Samples Demographics.csv", "a\n")
    counts = data.extract_required_files(zpath, tmp_path / "out")
    assert counts == {"tic_files": 1, "metadata_files": 1}
    assert not (tmp_path / "out" / "Boston" / "Boston" / "BST1.D" / "other.xml").exists()


def test_pick_samples_by_label_is_deterministic(synthetic_root):
    tic = data.build_tic_table(synthetic_root)
    picked = data.pick_samples_by_label(tic, n_each=2, random_state=42)
    assert picked == data.pick_samples_by_label(tic, n_each=2, random_state=42)
    assert len(picked[0]) == 2 and len(picked[1]) == 2 and not set(picked[0]) & set(picked[1])


def test_cancer_flag_handles_missing_and_text():
    assert data.cancer_flag(pd.Series({"cancer_binary": np.nan})) is None
    assert data.cancer_flag(pd.Series({"cancer_status": "Non-Cancer"})) == 0
    assert data.cancer_flag(pd.Series({"cancer_status": "Cancer"})) == 1
