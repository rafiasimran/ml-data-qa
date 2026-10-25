"""
Leakage: information in the features that will not exist at prediction time.

This is the family of defect that makes a model look better the more broken it
is, which is why every check here is a BLOCKER. A leak does not announce itself
as an error; it announces itself as an unusually good result.
"""
from __future__ import annotations

import pandas as pd

from ..config import ID_COLUMNS, TARGET, TIME_COLUMN
from ..severity import CheckResult, Severity
from ..stats import directionless_auc, target_rate_encode

# Above this share of distinct values, a column identifies rows rather than
# describing them. Target-rate encoding such a column produces a perfect score
# that says nothing, so these are reported separately instead.
ID_LIKE_CARDINALITY = 0.5


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in {TARGET, TIME_COLUMN, *ID_COLUMNS}]


def check_single_feature_leakage(train: pd.DataFrame, auc_threshold: float) -> CheckResult:
    """
    Score every feature against the target on its own.

    A single column that separates the classes almost perfectly is rarely a
    brilliant sensor. Usually it was written down after the outcome was known.
    """
    if TARGET not in train.columns:
        return CheckResult(
            check_id="LEAK-01", name="No single feature predicts the target", passed=False,
            severity=Severity.BLOCKER, summary="Target column absent", evidence={},
        )

    y = train[TARGET]
    scores: dict[str, float] = {}
    id_like: list[str] = []

    for col in _feature_columns(train):
        s = train[col]
        if s.nunique(dropna=False) <= 1:
            continue

        if pd.api.types.is_numeric_dtype(s):
            value = directionless_auc(s, y)
        else:
            if s.nunique() / max(len(s), 1) > ID_LIKE_CARDINALITY:
                id_like.append(col)
                continue
            value = directionless_auc(target_rate_encode(s, y), y)

        if not pd.isna(value):
            scores[col] = round(float(value), 4)

    suspects = {c: v for c, v in scores.items() if v >= auc_threshold}
    ranked = dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))

    return CheckResult(
        check_id="LEAK-01",
        name="No single feature predicts the target on its own",
        passed=not suspects,
        severity=Severity.BLOCKER,
        summary=(
            f"Strongest single feature reaches {max(scores.values()):.3f} AUC, below the "
            f"{auc_threshold} threshold" if not suspects and scores else
            f"{len(suspects)} feature(s) predict the target almost alone: " +
            ", ".join(f"{c} ({v:.3f} AUC)" for c, v in suspects.items())
        ),
        remedy="Ask when each flagged column gets its value. If it is written after the "
        "outcome is known, it cannot be a feature — it is the answer. Drop it and expect "
        "the model's score to fall sharply; the lower number is the real one.",
        evidence={
            "auc_by_feature": ranked,
            "suspects": suspects,
            "threshold": auc_threshold,
            "id_like_columns_skipped": id_like,
        },
    )


#: One class missing this many times more often than the other is a leak.
MISSINGNESS_RATE_RATIO = 2.0
#: Below this absolute gap the ratio is noise — 0.10% against 0.03% is 3x and means nothing.
MISSINGNESS_MIN_GAP_PCT = 2.0


def check_missingness_leakage(train: pd.DataFrame, auc_threshold: float) -> CheckResult:
    """
    Does the fact that a value is absent predict the target?

    This one hides from the usual checks. The column's values look innocent, its
    missing rate looks unremarkable, and imputing the mean appears to deal with
    it — while leaving the pattern that carried the leak fully intact.

    Judged on the RATIO of missing rates between the classes, not on AUC. AUC is
    the wrong statistic for a binary indicator: it is bounded by
    (sensitivity + specificity) / 2, so a sensor that drops out on 21% of bad
    parts and 3% of good ones — a sevenfold difference, and an obvious leak —
    scores only 0.59 and slips under any threshold set for a real feature. The
    ratio sees it immediately. AUC is still reported alongside, because it is
    what a reviewer will expect to see and its weakness here is worth showing.
    """
    if TARGET not in train.columns:
        return CheckResult(
            check_id="LEAK-02", name="Missingness does not predict the target", passed=False,
            severity=Severity.BLOCKER, summary="Target column absent", evidence={},
        )

    y = train[TARGET]
    findings: dict[str, dict] = {}
    suspects: dict[str, float] = {}

    for col in _feature_columns(train):
        is_missing = train[col].isna()
        if not is_missing.any() or is_missing.all():
            continue

        rate_pos = float(is_missing[y == 1].mean())
        rate_neg = float(is_missing[y == 0].mean())
        hi, lo = max(rate_pos, rate_neg), min(rate_pos, rate_neg)

        # Guard the division: an unobserved rate of zero is an infinite ratio.
        ratio = hi / lo if lo > 0 else float("inf")
        gap_pct = 100 * (hi - lo)

        findings[col] = {
            "missing_rate_when_positive": round(rate_pos, 4),
            "missing_rate_when_negative": round(rate_neg, 4),
            "rate_ratio": round(ratio, 2) if ratio != float("inf") else "inf",
            "gap_pct_points": round(gap_pct, 2),
            "auc_of_indicator": round(float(directionless_auc(is_missing.astype(int), y)), 4),
        }

        if ratio >= MISSINGNESS_RATE_RATIO and gap_pct >= MISSINGNESS_MIN_GAP_PCT:
            suspects[col] = ratio

    return CheckResult(
        check_id="LEAK-02",
        name="Missingness does not predict the target",
        passed=not suspects,
        severity=Severity.BLOCKER,
        summary=(
            "No column's missingness pattern predicts the target" if not suspects else
            f"{len(suspects)} column(s) leak through their missingness: " +
            ", ".join(
                f"{c} (missing {findings[c]['rate_ratio']}x more often in one class)"
                for c in suspects
            )
        ),
        remedy="Imputing will not fix this — the pattern survives imputation and the model "
        "picks it up through the missing-value indicator. Find out why the sensor drops out "
        "more often for one class. If the dropout is caused by the outcome rather than "
        "happening to coincide with it, the column cannot be used at all.",
        evidence={
            "by_column": findings,
            "suspects": {c: findings[c]["rate_ratio"] for c in suspects},
            "criteria": {
                "rate_ratio_at_or_above": MISSINGNESS_RATE_RATIO,
                "and_gap_pct_points_at_or_above": MISSINGNESS_MIN_GAP_PCT,
                "auc_threshold_reported_but_not_used": auc_threshold,
            },
        },
    )


def check_id_like_columns(train: pd.DataFrame) -> CheckResult:
    """High-cardinality identifiers passed to a model memorise rather than generalise."""
    flagged = {}
    for col in _feature_columns(train):
        s = train[col]
        if pd.api.types.is_numeric_dtype(s):
            continue
        ratio = s.nunique() / max(len(s), 1)
        if ratio > ID_LIKE_CARDINALITY:
            flagged[col] = round(float(ratio), 4)

    return CheckResult(
        check_id="LEAK-03",
        name="No identifier-like columns among the features",
        passed=not flagged,
        severity=Severity.MINOR,
        summary="No identifier-like feature columns" if not flagged else
        f"{len(flagged)} near-unique column(s) sitting in the feature set: {', '.join(flagged)}",
        remedy="A near-unique column lets the model memorise individual rows. Exclude it "
        "from the features, or replace it with the attribute you actually meant to capture.",
        evidence={"distinct_value_ratio": flagged},
    )
