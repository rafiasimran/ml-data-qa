"""
Integrity of the split itself.

These are the checks that decide whether a reported test score means anything.
Everything else in the suite affects how good the model is; these affect whether
you can believe the number at all.
"""
from __future__ import annotations

import pandas as pd

from ..config import ID_COLUMNS, TIME_COLUMN
from ..severity import CheckResult, Severity


def _row_hashes(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """
    A hash per row, for comparing row content across two frames.

    Uses pandas' own row hashing rather than joining the values into a string.
    Joining breaks the moment a column holds a NaN beside text, and it costs a
    Python-level pass over every row; hashing is vectorised and handles mixed
    dtypes and nulls natively. Columns are taken in a fixed order so the two
    frames hash the same row to the same value.
    """
    return pd.util.hash_pandas_object(df[columns], index=False)


def check_duplicate_rows(train: pd.DataFrame, max_pct: float) -> CheckResult:
    n_dupes = int(train.duplicated().sum())
    pct = round(100 * n_dupes / max(len(train), 1), 2)
    return CheckResult(
        check_id="DUP-01",
        name="No duplicate rows within train",
        passed=pct <= max_pct,
        severity=Severity.MAJOR,
        summary=f"{n_dupes:,} exact duplicate row(s) in train ({pct}%)",
        remedy="Duplicates make the model confident about whatever it has seen twice and "
        "inflate the effective sample size. Deduplicate before splitting, not after, "
        "otherwise copies of one row can end up on both sides.",
        evidence={"duplicate_rows": n_dupes, "duplicate_pct": pct},
    )


def check_duplicate_ids(train: pd.DataFrame) -> CheckResult:
    present = [c for c in ID_COLUMNS if c in train.columns]
    if not present:
        return CheckResult(
            check_id="DUP-02", name="Identifiers are unique", passed=True,
            severity=Severity.MINOR, summary="No identifier column configured",
            evidence={},
        )
    col = present[0]
    n = int(train[col].duplicated().sum())
    return CheckResult(
        check_id="DUP-02",
        name="Identifiers are unique",
        passed=n == 0,
        severity=Severity.MAJOR,
        summary=f"{n:,} repeated value(s) in '{col}'" if n else f"'{col}' is unique",
        remedy="A repeated identifier means either a duplicate export or an identifier "
        "that does not identify what you assumed. Find out which before modelling.",
        evidence={"id_column": col, "repeated": n},
    )


def check_train_test_overlap(
    train: pd.DataFrame, test: pd.DataFrame, max_overlap: int
) -> CheckResult:
    """
    The single most damaging defect in this suite.

    If rows appear on both sides, the test set is not unseen data. Every metric
    computed on it is optimistic, and the amount of optimism is unknowable, which
    is worse than a known bias.
    """
    present_ids = [c for c in ID_COLUMNS if c in train.columns and c in test.columns]
    evidence: dict = {}

    if present_ids:
        col = present_ids[0]
        shared = set(train[col]) & set(test[col])
        n_id_overlap = len(shared)
        evidence["shared_id_values"] = n_id_overlap
        evidence["id_column"] = col
        evidence["examples"] = sorted(list(shared))[:5]
    else:
        n_id_overlap = 0

    # Also compare full row content, which catches an overlap that was copied
    # with a fresh identifier attached.
    shared_cols = [c for c in train.columns if c in test.columns and c not in ID_COLUMNS]
    if shared_cols:
        tr = _row_hashes(train, shared_cols)
        te = _row_hashes(test, shared_cols)
        n_content_overlap = int(te.isin(set(tr)).sum())
    else:
        n_content_overlap = 0
    evidence["identical_content_rows"] = n_content_overlap

    worst = max(n_id_overlap, n_content_overlap)
    return CheckResult(
        check_id="SPLIT-01",
        name="Train and test do not overlap",
        passed=worst <= max_overlap,
        severity=Severity.BLOCKER,
        summary=(
            "No overlap between train and test" if worst == 0 else
            f"{worst:,} row(s) appear in both train and test"
        ),
        remedy="Rebuild the split from deduplicated data, grouping by the real unit of "
        "observation. Until then no score measured on this test set means anything, "
        "and a higher score means more contamination rather than a better model.",
        evidence=evidence,
    )


def check_temporal_split(train: pd.DataFrame, test: pd.DataFrame) -> CheckResult:
    """
    Was the split made across time, or through it?

    For data produced in sequence, a random split trains the model on the future
    and tests it on the past. It will score well and then degrade on deployment,
    because production only ever offers it the past.
    """
    if TIME_COLUMN not in train.columns or TIME_COLUMN not in test.columns:
        return CheckResult(
            check_id="SPLIT-02", name="Split respects time order", passed=True,
            severity=Severity.MAJOR, summary="No time column configured", evidence={},
        )

    tr = pd.to_datetime(train[TIME_COLUMN], errors="coerce").dropna()
    te = pd.to_datetime(test[TIME_COLUMN], errors="coerce").dropna()
    if tr.empty or te.empty:
        return CheckResult(
            check_id="SPLIT-02", name="Split respects time order", passed=True,
            severity=Severity.MAJOR, summary="Time column unparseable", evidence={},
        )

    # Share of test rows dated before the last training row.
    overlap_share = float((te < tr.max()).mean())
    passed = overlap_share < 0.05

    return CheckResult(
        check_id="SPLIT-02",
        name="Split respects time order",
        passed=passed,
        severity=Severity.MAJOR,
        summary=(
            "Test period follows the training period" if passed else
            f"{overlap_share:.0%} of test rows are older than the end of the training "
            "period, so the split is random, not by time"
        ),
        remedy="Split on time: train on the earlier period, test on the later one. A random "
        "split answers 'can the model interpolate within a period it has seen', which is "
        "not the question production asks.",
        evidence={
            "train_range": [str(tr.min()), str(tr.max())],
            "test_range": [str(te.min()), str(te.max())],
            "test_rows_before_train_end_pct": round(100 * overlap_share, 2),
        },
    )
