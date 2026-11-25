from dataclasses import replace

import numpy as np
import pytest

from msm_olfactory.config import Settings
from msm_olfactory.synthetic import make_synthetic_dataset


@pytest.fixture
def fast_settings() -> Settings:
    """Original model specification with a lighter start-value search and 1 training epoch, for speed."""
    return replace(Settings(), search_reps=3, search_iter=10, em_iter=5, station_step=60, epochs=1, workers=1)


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory):
    """A small SYNTHETIC dataset (2 sites x 6 samples x 500 points) in the real folder layout."""
    return make_synthetic_dataset(tmp_path_factory.mktemp("synthetic"), n_per_site=6, n_points=500, seed=3)


@pytest.fixture
def two_regime_series():
    """A series that switches between two well-separated mean levels with persistent regimes."""
    rng = np.random.default_rng(0)
    states, s = [], 0
    for _ in range(600):
        if rng.random() < 0.02:
            s = 1 - s
        states.append(s)
    states = np.array(states)
    return np.where(states == 1, 3.0, -3.0) + rng.normal(0, 0.5, len(states)), states
