"""Structural checks: is this even the dataset we think it is?"""
from __future__ import annotations

import pandas as pd

from ..config import EXPECTED_SCHEMA, PLAUSIBLE_RANGES
from ..severity import CheckResult, Severity


def _kind(dtype) -> str:
    if pd.api.types.is_integer_dtype(dtype):
        return "int"
    if pd.api.types.is_float_dtype(dtype):
        return "float"
    if pd.api.types.is_bool_dtype(dtype):
        return "bool"
    return "object"


def check_expected_columns(train: pd.DataFrame) -> CheckResult:
    missing = sorted(set(EXPECTED_SCHEMA) - set(train.columns))
    unexpected = sorted(set(train.columns) - set(EXPECTED_SCHEMA))
    passed = not missing
    return CheckResult(
        check_id="SCHEMA-01",
        name="Expected columns are present",
        passed=passed,
        severity=Severity.BLOCKER,
        summary=(
            "All expected columns present"
            + (f"; {len(unexpected)} extra column(s) not in the contract" if unexpected else "")
            if passed
            else f"{len(missing)} expected column(s) missing: {', '.join(missing)}"
        ),
        remedy=(
            "Extra columns are not automatically wrong, but each one needs a decision: "
            "feature, identifier, or leak. Check them against the leakage results."
            if passed
            else "The export is not the dataset the pipeline was written for. Fix the "
            "export rather than adapting the code to whatever arrived."
        ),
        evidence={"missing": missing, "unexpected": unexpected},
    )


def check_column_types(train: pd.DataFrame) -> CheckResult:
    wrong = {
        col: {"expected": want, "actual": _kind(train[col].dtype)}
        for col, want in EXPECTED_SCHEMA.items()
        if col in train.columns and _kind(train[col].dtype) != want
    }
    return CheckResult(
        check_id="SCHEMA-02",
        name="Column types match the contract",
        passed=not wrong,
        severity=Severity.MAJOR,
        summary="All column types as expected" if not wrong else f"{len(wrong)} column(s) typed unexpectedly",
        remedy="A numeric column read as text usually means a stray non-numeric value "
        "in the export. Find that value; do not silently coerce the column.",
        evidence={"mismatches": wrong},
    )


def check_train_test_same_columns(train: pd.DataFrame, test: pd.DataFrame) -> CheckResult:
    only_train = sorted(set(train.columns) - set(test.columns))
    only_test = sorted(set(test.columns) - set(train.columns))
    passed = not only_train and not only_test
    return CheckResult(
        check_id="SCHEMA-03",
        name="Train and test share the same columns",
        passed=passed,
        severity=Severity.BLOCKER,
        summary="Train and test columns match" if passed else
        f"{len(only_train)} column(s) only in train, {len(only_test)} only in test",
        remedy="A column present at training time and absent at scoring time will fail "
        "in production, whatever the offline metrics say.",
        evidence={"only_in_train": only_train, "only_in_test": only_test},
    )


def check_plausible_ranges(train: pd.DataFrame) -> CheckResult:
    violations: dict[str, dict] = {}
    for col, (lo, hi) in PLAUSIBLE_RANGES.items():
        if col not in train.columns:
            continue
        s = pd.to_numeric(train[col], errors="coerce")
        bad = ((s < lo) | (s > hi)).fillna(False)
        if bad.any():
            violations[col] = {
                "count": int(bad.sum()),
                "allowed": [lo, hi],
                "observed_min": float(s.min()),
                "observed_max": float(s.max()),
            }
    total = sum(v["count"] for v in violations.values())
    return CheckResult(
        check_id="RANGE-01",
        name="Values are physically possible",
        passed=not violations,
        severity=Severity.MAJOR,
        summary="All values within plausible ranges" if not violations else
        f"{total:,} value(s) outside plausible ranges across {len(violations)} column(s)",
        remedy="These are sensor faults, not measurements. Treat them as missing rather "
        "than clipping them to the limit, which would invent a reading that was never taken.",
        evidence={"violations": violations},
    )


def check_constant_columns(train: pd.DataFrame) -> CheckResult:
    constant = [
        c for c in train.columns
        if train[c].nunique(dropna=False) <= 1
    ]
    return CheckResult(
        check_id="SCHEMA-04",
        name="No constant columns",
        passed=not constant,
        severity=Severity.MINOR,
        summary="No constant columns" if not constant else
        f"{len(constant)} constant column(s): {', '.join(constant)}",
        remedy="A column with one value carries no information. Harmless to a tree model, "
        "but it inflates the feature count and hides that the field was never populated.",
        evidence={"constant_columns": constant},
    )


def check_missingness(train: pd.DataFrame, max_missing_pct: float) -> CheckResult:
    pct = (train.isna().mean() * 100).round(2)
    over = {c: float(p) for c, p in pct.items() if p > max_missing_pct}
    return CheckResult(
        check_id="NULL-01",
        name="Missing value rates are within limits",
        passed=not over,
        severity=Severity.MINOR,
        summary=f"No column above {max_missing_pct}% missing" if not over else
        f"{len(over)} column(s) above {max_missing_pct}% missing",
        remedy="Decide per column whether to impute or drop, and record the decision. "
        "Check the missingness leakage result before imputing anything.",
        evidence={"missing_pct": {c: float(p) for c, p in pct.items() if p > 0}},
    )
