"""Small shared helpers: logging setup, JSON serialisation and quieting third-party console output."""

from __future__ import annotations

import contextlib
import io
import json
import logging
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

LOG_FORMAT = "%(asctime)s  %(levelname)-7s %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Send package log messages to the console in a concise format (used by the command-line scripts)."""
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt="%H:%M:%S", force=True)
    # Third-party libraries log at INFO/DEBUG; keep the console focused on pipeline progress.
    for noisy in ("matplotlib", "PIL", "numexpr"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@contextlib.contextmanager
def quiet_numerics() -> Iterator[None]:
    """Silence statsmodels/numpy warnings during model fitting.

    The source notebook suppressed all warnings globally. Convergence and EM re-scaling warnings from the MSM fits, and
    KPSS p-value interpolation warnings, are expected here and are reflected in the recorded fit statistics.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@contextlib.contextmanager
def suppress_stdout() -> Iterator[None]:
    """Discard stdout printed by third-party code (``neuroCombat`` prints its progress unconditionally)."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def to_jsonable(obj: Any) -> Any:
    """Convert numpy scalars/arrays and non-finite floats (to ``None``) so the object can be written as JSON."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    return obj


def write_json(path: Path, document: Any) -> Path:
    """Write ``document`` as indented JSON, creating parent folders as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(document), indent=2) + "\n", encoding="utf-8")
    return path
