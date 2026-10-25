"""
Does the test set come from the same population as the training set?

A model fitted on one distribution and scored on another gives a number that
describes neither. This matters twice: between train and test now, and between
training data and production later.
"""
from __future__ import annotations

import pandas as pd

from ..config import ID_COLUMNS, TARGET, TIME_COLUMN
from ..severity import CheckResult, Severity
from ..stats import psi


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in {TARGET, TIME_COLUMN, *ID_COLUMNS}]


def check_numeric_drift(
    train: pd.DataFrame, test: pd.DataFrame, warn: float, fail: float
) -> CheckResult:
    scores: dict[str, float] = {}
    for col in _feature_columns(train):
        if col not in test.columns or not pd.api.types.is_numeric_dtype(train[col]):
            continue
        value = psi(train[col], test[col])
        if not pd.isna(value):
            scores[col] = round(float(value), 4)

    failing = {c: v for c, v in scores.items() if v >= fail}
    warning = {c: v for c, v in scores.items() if warn <= v < fail}
    ranked = dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))

    return CheckResult(
        check_id="DRIFT-01",
        name="Numeric features have the same distribution in train and test",
        passed=not failing,
        severity=Severity.MAJOR,
        summary=(
            (f"No feature exceeds PSI {fail}" if not warning else
             f"No feature exceeds PSI {fail}; {len(warning)} in the warning band")
            if not failing else
            f"{len(failing)} feature(s) shifted beyond PSI {fail}: " +
            ", ".join(f"{c} (PSI {v})" for c, v in failing.items())
        ),
        remedy="A shifted feature usually means a recalibrated instrument, a process change, "
        "or a split made across a boundary where something changed. Find the cause before "
        "modelling: the model will learn the pre-change relationship and be scored on the "
        "post-change one.",
        evidence={"psi_by_feature": ranked, "failing": failing, "warning": warning,
                  "thresholds": {"warn": warn, "fail": fail}},
    )


def check_categorical_drift(train: pd.DataFrame, test: pd.DataFrame) -> CheckResult:
    """Categories that only appear on one side will break the encoder later."""
    problems: dict[str, dict] = {}
    for col in _feature_columns(train):
        if col not in test.columns or pd.api.types.is_numeric_dtype(train[col]):
            continue
        tr_vals, te_vals = set(train[col].dropna()), set(test[col].dropna())
        unseen = sorted(te_vals - tr_vals)
        absent = sorted(tr_vals - te_vals)
        if unseen or absent:
            problems[col] = {
                "in_test_not_train": unseen[:10],
                "in_train_not_test": absent[:10],
                "n_unseen": len(unseen),
            }

    unseen_total = sum(p["n_unseen"] for p in problems.values())
    return CheckResult(
        check_id="DRIFT-02",
        name="Categorical levels are consistent across train and test",
        passed=unseen_total == 0,
        severity=Severity.MAJOR,
        summary=(
            "All categorical levels appear in both splits" if not problems else
            f"{unseen_total} level(s) appear in test but never in train"
        ),
        remedy="A level the model never saw in training has no learned behaviour. Decide "
        "now how unseen levels are handled (e.g. an 'unknown' bucket) rather than "
        "discovering it when the encoder raises in production.",
        evidence={"columns": problems},
    )


def check_target_drift(train: pd.DataFrame, test: pd.DataFrame) -> CheckResult:
    if TARGET not in train.columns or TARGET not in test.columns:
        return CheckResult(
            check_id="DRIFT-03", name="Target rate is stable across splits", passed=True,
            severity=Severity.MAJOR, summary="Target absent from one split", evidence={},
        )
    tr_rate = float(train[TARGET].mean())
    te_rate = float(test[TARGET].mean())
    # Relative change, so the threshold means the same thing at any base rate.
    rel = abs(te_rate - tr_rate) / max(tr_rate, 1e-9)
    passed = rel < 0.20
    return CheckResult(
        check_id="DRIFT-03",
        name="Target rate is stable across splits",
        passed=passed,
        severity=Severity.MAJOR,
        summary=(
            f"Positive rate {tr_rate:.2%} in train, {te_rate:.2%} in test "
            f"({rel:.0%} relative difference)"
        ),
        remedy="A different base rate on each side shifts every threshold-dependent metric. "
        "If the split was random this points at an underlying change over time; if it was "
        "chronological, the process itself moved.",
        evidence={"train_positive_rate": round(tr_rate, 4),
                  "test_positive_rate": round(te_rate, 4),
                  "relative_difference": round(rel, 4)},
    )


def check_temporal_drift(
    train: pd.DataFrame, test: pd.DataFrame, warn: float, fail: float
) -> CheckResult:
    """
    Compare the earliest period of the data against the latest, ignoring the split.

    Train vs test only tells you whether the two halves match. With a random
    split they always will, because a change in the middle of the period ends
    up in both halves equally. Production only sees the newest data though, so
    this check puts the data back together, sorts it by time and compares the
    first 70% with the last 30%.
    """
    if TIME_COLUMN not in train.columns:
        return CheckResult(
            check_id="DRIFT-04", name="Features are stable over time", passed=True,
            severity=Severity.MAJOR, summary="No time column configured", evidence={},
        )

    combined = pd.concat([train, test], ignore_index=True)
    combined["_ts"] = pd.to_datetime(combined[TIME_COLUMN], errors="coerce")
    combined = combined.dropna(subset=["_ts"]).sort_values("_ts")
    if len(combined) < 200:
        return CheckResult(
            check_id="DRIFT-04", name="Features are stable over time", passed=True,
            severity=Severity.MAJOR, summary="Too few rows to assess", evidence={},
        )

    cut = int(len(combined) * 0.70)
    early, late = combined.iloc[:cut], combined.iloc[cut:]

    scores: dict[str, float] = {}
    for col in _feature_columns(combined):
        if col == "_ts" or not pd.api.types.is_numeric_dtype(combined[col]):
            continue
        value = psi(early[col], late[col])
        if not pd.isna(value):
            scores[col] = round(float(value), 4)

    failing = {c: v for c, v in scores.items() if v >= fail}
    ranked = dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))

    return CheckResult(
        check_id="DRIFT-04",
        name="Features are stable over time",
        passed=not failing,
        severity=Severity.MAJOR,
        summary=(
            f"No feature drifts beyond PSI {fail} across the period" if not failing else
            f"{len(failing)} feature(s) drift across the period: " +
            ", ".join(f"{c} (PSI {v})" for c, v in failing.items())
        ),
        remedy="This drift is invisible to a train-versus-test comparison when the split "
        "was random, because the change lands on both halves equally. It is still real, and "
        "production only sees the later period. Find what changed (a recalibrated "
        "instrument, a new supplier, a process change) and decide whether the earlier "
        "data should be used for training at all.",
        evidence={
            "psi_by_feature_early_vs_late": ranked,
            "failing": failing,
            "early_period": [str(early["_ts"].min()), str(early["_ts"].max())],
            "late_period": [str(late["_ts"].min()), str(late["_ts"].max())],
            "note": "Compares the first 70% of the period against the last 30%, "
                    "independently of how train and test were split.",
        },
    )
