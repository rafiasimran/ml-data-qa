import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _clean_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """A small, deliberately well-behaved dataset. Every test starts from this
    and breaks exactly one thing, so a failure names its own cause."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-01-01", periods=n, freq="10min")
    return pd.DataFrame({
        "part_id": [f"P{i:05d}" for i in range(n)],
        "produced_at": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "line": rng.choice(["LINE-A", "LINE-B"], n),
        "shift": rng.choice(["early", "late"], n),
        "tool_id": rng.choice(["T-01", "T-02", "T-03"], n),
        "tool_wear_min": rng.uniform(0, 400, n).round(1),
        "temp_c": rng.normal(212, 6, n).round(2),
        "pressure_bar": rng.normal(88, 3, n).round(2),
        "vibration_mm_s": rng.normal(0.42, 0.08, n).round(4),
        "cycle_time_s": rng.normal(31, 2, n).round(2),
        "humidity_pct": rng.normal(44, 7, n).round(1),
        "defect": rng.integers(0, 2, n),
    })


@pytest.fixture
def clean_train() -> pd.DataFrame:
    return _clean_frame(400, seed=1)


@pytest.fixture
def clean_test() -> pd.DataFrame:
    return _clean_frame(150, seed=99).assign(
        part_id=lambda d: ["Q" + p[1:] for p in d.part_id]
    )
