"""Checks on the target column itself."""
from __future__ import annotations

import pandas as pd

from ..config import ID_COLUMNS, TARGET, TIME_COLUMN
from ..severity import CheckResult, Severity
from ..stats import knn_label_disagreement


def check_target_present(train: pd.DataFrame) -> CheckResult:
    present = TARGET in train.columns
    n_null = int(train[TARGET].isna().sum()) if present else 0
    return CheckResult(
        check_id="LABEL-01",
        name="Target column is present and complete",
        passed=present and n_null == 0,
        severity=Severity.BLOCKER,
        summary=(
            f"Target '{TARGET}' missing from the dataset" if not present
            else f"{n_null:,} row(s) have no label" if n_null
            else f"Target '{TARGET}' present with no missing values"
        ),
        remedy="Rows with no label cannot be trained on. Find out whether they are "
        "genuinely unlabelled or whether the join that attached labels dropped them.",
        evidence={"present": present, "null_labels": n_null},
    )


def check_target_values(train: pd.DataFrame) -> CheckResult:
    if TARGET not in train.columns:
        return CheckResult(
            check_id="LABEL-02", name="Target takes only expected values", passed=False,
            severity=Severity.BLOCKER, summary="Target column absent", evidence={},
        )
    values = sorted(pd.Series(train[TARGET].dropna().unique()).tolist())
    unexpected = [v for v in values if v not in (0, 1)]
    return CheckResult(
        check_id="LABEL-02",
        name="Target takes only expected values",
        passed=not unexpected,
        severity=Severity.BLOCKER,
        summary=f"Target values: {values}" if not unexpected else
        f"Unexpected target value(s): {unexpected}",
        remedy="A third value usually means 'not yet inspected' encoded as a label. "
        "Those rows are unlabelled, not negative.",
        evidence={"observed_values": [str(v) for v in values]},
    )


def check_class_balance(train: pd.DataFrame, min_share_pct: float) -> CheckResult:
    if TARGET not in train.columns:
        return CheckResult(
            check_id="LABEL-03", name="Classes are usably balanced", passed=False,
            severity=Severity.MAJOR, summary="Target column absent", evidence={},
        )
    counts = train[TARGET].value_counts(dropna=False)
    shares = (counts / counts.sum() * 100).round(2)
    minority = float(shares.min())
    passed = minority >= min_share_pct

    majority_share = float(shares.max())
    return CheckResult(
        check_id="LABEL-03",
        name="Classes are usably balanced",
        passed=passed,
        severity=Severity.MAJOR,
        summary=(
            f"Minority class is {minority}% of rows" +
            ("" if passed else f" — below the {min_share_pct}% floor")
        ),
        remedy=(
            f"Accuracy is the wrong metric here: predicting the majority class for every "
            f"row scores {majority_share:.1f}%. Report precision, recall and AUC, and "
            f"stratify the split so both classes appear on each side."
        ),
        evidence={
            "class_counts": {str(k): int(v) for k, v in counts.items()},
            "class_share_pct": {str(k): float(v) for k, v in shares.items()},
            "accuracy_of_predicting_majority": round(majority_share, 2),
        },
    )


def check_label_noise(train: pd.DataFrame) -> CheckResult:
    """
    Estimate how often a row's label disagrees with its neighbours.

    This cannot prove a label is wrong — a genuinely unusual part looks the same
    as a mislabelled one. It reports a rate so a human can decide whether the
    inspection process needs looking at.
    """
    if TARGET not in train.columns:
        return CheckResult(
            check_id="LABEL-04", name="Label noise is within reason", passed=True,
            severity=Severity.MINOR, summary="Target column absent", evidence={},
        )
    features = train.drop(columns=[c for c in [TARGET, TIME_COLUMN, *ID_COLUMNS] if c in train.columns])
    rate, n = knn_label_disagreement(features, train[TARGET])

    if pd.isna(rate):
        return CheckResult(
            check_id="LABEL-04", name="Label noise is within reason", passed=True,
            severity=Severity.MINOR, summary="Not enough numeric features to estimate",
            evidence={},
        )

    pct = round(100 * rate, 2)
    passed = pct < 5.0
    return CheckResult(
        check_id="LABEL-04",
        name="Label noise is within reason",
        passed=passed,
        severity=Severity.MINOR,
        summary=f"About {pct}% of sampled rows carry a label their nearest neighbours contradict",
        remedy="Estimated on a sample, and a surprising label is not automatically a wrong "
        "one. Treat it as a prompt to review the inspection process, not a list to delete. "
        "It also puts a ceiling on achievable accuracy that is worth knowing before "
        "chasing the last few points.",
        evidence={"suspect_pct": pct, "suspect_rows_in_sample": n, "method": "k=10 nearest neighbours, 1500-row sample"},
    )
