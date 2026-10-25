"""
Thresholds, in one place and named.

Every number here is a judgement call, so each one carries the reasoning. A
threshold with no stated reason gets tuned until the suite stops complaining,
which is how a quality gate becomes decoration.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

TARGET = "defect"
ID_COLUMNS = ["part_id"]
TIME_COLUMN = "produced_at"

EXPECTED_SCHEMA: dict[str, str] = {
    "part_id": "object",
    "produced_at": "object",
    "line": "object",
    "shift": "object",
    "tool_id": "object",
    "tool_wear_min": "float",
    "temp_c": "float",
    "pressure_bar": "float",
    "vibration_mm_s": "float",
    "cycle_time_s": "float",
    "humidity_pct": "float",
    "defect": "int",
}

# Physically possible ranges, from the process rather than from the data.
# Deriving limits from the data would accept whatever junk is already in it.
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "temp_c": (120.0, 320.0),
    "pressure_bar": (30.0, 140.0),
    "vibration_mm_s": (0.0, 3.0),
    "cycle_time_s": (5.0, 120.0),
    "humidity_pct": (0.0, 100.0),
    "tool_wear_min": (0.0, 600.0),
}


@dataclass(frozen=True)
class Thresholds:
    # A single feature that predicts the target this well on its own is not a
    # feature, it is the answer written in another column. 0.80 is low enough
    # to catch a leak and high enough that a genuinely strong sensor passes.
    leakage_auc: float = float(os.getenv("LEAKAGE_AUC", "0.80"))

    # Whether a value is MISSING should not predict the target. If it does,
    # the missingness pattern is itself a leak and imputing hides it.
    missingness_auc: float = float(os.getenv("MISSINGNESS_AUC", "0.60"))

    # Population Stability Index. The industry convention: under 0.10 stable,
    # 0.10-0.25 moderate shift, above 0.25 the populations differ enough that
    # a model fitted on one will misbehave on the other.
    psi_warn: float = float(os.getenv("PSI_WARN", "0.10"))
    psi_fail: float = float(os.getenv("PSI_FAIL", "0.25"))

    # Below this the minority class is too rare for accuracy to mean anything
    # and the split needs stratifying.
    min_class_share_pct: float = float(os.getenv("MIN_CLASS_SHARE_PCT", "5.0"))

    # Exact duplicates are never acceptable in volume; a few can be genuine.
    max_duplicate_pct: float = float(os.getenv("MAX_DUPLICATE_PCT", "0.5"))

    # Any overlap at all between train and test is a blocker. There is no
    # tolerable amount of answer key in the exam.
    max_train_test_overlap: int = 0

    max_missing_pct: float = float(os.getenv("MAX_MISSING_PCT", "10.0"))

    reports_dir: Path = field(
        default_factory=lambda: Path(os.getenv("REPORTS_DIR", REPO_ROOT / "reports"))
    )


thresholds = Thresholds()
