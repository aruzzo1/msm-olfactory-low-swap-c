"""SYNTHETIC demo data that mimics the file layout of the real dataset, for running the pipeline without it.

!!! These chromatograms and "clinical" rows are randomly generated. They are not patient data, are not derived from
the Rotteveel et al. dataset, and labels are assigned at random, independently of the signals. Any classification
score obtained on them is meaningless and must never be reported as a result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from msm_olfactory.config import METADATA_FILENAME, TIC_FILENAME

SYNTHETIC_HEADER = "SYNTHETIC DEMO DATA - randomly generated, not real GC-MS or patient data"
SITES = {"Boston": "BST", "Duke": "DK"}

#: Label used on every instructional figure drawn from synthetic data.
ILLUSTRATIVE_LABEL = "Synthetic illustrative example"
ILLUSTRATIVE_SEED = 2026
ILLUSTRATIVE_POINTS = 3000


def synthetic_tic(rng: np.random.Generator, n_points: int, site_offset: float) -> tuple[np.ndarray, np.ndarray]:
    """Baseline + slow drift + random Gaussian peaks + noise, on a 1-40 min retention-time axis."""
    time = np.linspace(1.09, 40.0, n_points)
    baseline = 8e5 * (1 + site_offset) + 2e4 * (time - time.mean())
    peaks = np.zeros(n_points)
    for _ in range(rng.integers(15, 30)):
        centre, width, height = rng.uniform(3, 38), rng.uniform(0.05, 0.3), rng.lognormal(15, 1.0)
        peaks += height * np.exp(-0.5 * ((time - centre) / width) ** 2)
    noise = rng.normal(0, 2e4, n_points)
    return time, np.clip(baseline + peaks + noise, 0, None)


def make_synthetic_dataset(out_dir: Path, n_per_site: int = 12, n_points: int = 1200, seed: int = 7) -> Path:
    """Write ``<out_dir>/<site>/<site>/<ID>.D/tic_back.csv`` files and a matching demographics CSV."""
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    rows = []
    for site_idx, (site, prefix) in enumerate(SITES.items()):
        for k in range(n_per_site):
            sample_id = f"{prefix}{9000 + k}"
            time, intensity = synthetic_tic(rng, n_points, site_offset=0.35 * site_idx)
            folder = out_dir / site / site / f"{sample_id}.D"
            folder.mkdir(parents=True, exist_ok=True)
            with open(folder / TIC_FILENAME, "w", encoding="utf-8", newline="\n") as f:
                f.write(f"{SYNTHETIC_HEADER}\nStart of data points\n")
                for t, v in zip(time, intensity, strict=True):
                    f.write(f"{t:.5f},{v:.0f}\n")
            cancer = int(rng.integers(0, 2))  # random label: no relationship to the signal
            rows.append(
                {
                    "S/N": len(rows) + 1,
                    "Source": site,
                    "Samples": sample_id,
                    "prostate.cancer": cancer,
                    "Status": "Cancer" if cancer else "Non-Cancer",
                    "Class/Gleason Score": "6" if cancer else "BPH",
                    "Risk Level": "Low-Risk" if cancer else "Control",
                    "Age": int(rng.integers(50, 80)),
                    "PSA (ng/ml)": round(float(rng.uniform(0.5, 10)), 1),
                    "Race": "SYNTHETIC",
                }
            )
    # guarantee both classes at every site so stratified splits and ComBat are always possible
    df = pd.DataFrame(rows)
    for site in SITES:
        idx = df.index[df["Source"] == site]
        df.loc[idx[0], ["prostate.cancer", "Status", "Class/Gleason Score", "Risk Level"]] = [
            1,
            "Cancer",
            "6",
            "Low-Risk",
        ]
        df.loc[idx[1], ["prostate.cancer", "Status", "Class/Gleason Score", "Risk Level"]] = [
            0,
            "Non-Cancer",
            "BPH",
            "Control",
        ]
    df.to_csv(out_dir / METADATA_FILENAME, index=False)
    (out_dir / "README_SYNTHETIC.txt").write_text(SYNTHETIC_HEADER + "\n", encoding="utf-8")
    return out_dir


def make_illustrative_tic(
    n_points: int = ILLUSTRATIVE_POINTS, seed: int = ILLUSTRATIVE_SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Return one SYNTHETIC chromatogram ``(retention_time, intensity)`` for instructional figures.

    Used instead of a real sample to illustrate the MSM regimes, the rolling stationarity tests and the TIC image, so no
    source data appears in published figures. It is not data and must not be presented as an experimental result.
    """
    return synthetic_tic(np.random.default_rng(seed), n_points, site_offset=0.0)
