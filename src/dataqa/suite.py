"""
Run every check and assemble the report.

The order is the order a person would investigate in: is the dataset the right
shape, is the split honest, is the target sound, is anything leaking, do the two
halves match. A structural failure makes the later results meaningless, so it
should be read first.
"""
from __future__ import annotations

import logging

import pandas as pd

from .checks import drift, integrity, labels, leakage, schema
from .config import Thresholds, thresholds as default_thresholds
from .severity import QAReport

log = logging.getLogger(__name__)


def run_suite(
    train: pd.DataFrame,
    test: pd.DataFrame,
    dataset_name: str = "dataset",
    t: Thresholds | None = None,
) -> QAReport:
    t = t or default_thresholds
    report = QAReport(dataset_name=dataset_name)
    report.row_counts = {
        "train_rows": len(train),
        "test_rows": len(test),
        "columns": len(train.columns),
    }

    checks = [
        # Structure
        lambda: schema.check_expected_columns(train),
        lambda: schema.check_column_types(train),
        lambda: schema.check_train_test_same_columns(train, test),
        lambda: schema.check_constant_columns(train),
        lambda: schema.check_plausible_ranges(train),
        lambda: schema.check_missingness(train, t.max_missing_pct),
        # Split integrity
        lambda: integrity.check_train_test_overlap(train, test, t.max_train_test_overlap),
        lambda: integrity.check_temporal_split(train, test),
        lambda: integrity.check_duplicate_rows(train, t.max_duplicate_pct),
        lambda: integrity.check_duplicate_ids(train),
        # Target
        lambda: labels.check_target_present(train),
        lambda: labels.check_target_values(train),
        lambda: labels.check_class_balance(train, t.min_class_share_pct),
        lambda: labels.check_label_noise(train),
        # Leakage
        lambda: leakage.check_single_feature_leakage(train, t.leakage_auc),
        lambda: leakage.check_missingness_leakage(train, t.missingness_auc),
        lambda: leakage.check_id_like_columns(train),
        # Drift
        lambda: drift.check_numeric_drift(train, test, t.psi_warn, t.psi_fail),
        lambda: drift.check_categorical_drift(train, test),
        lambda: drift.check_temporal_drift(train, test, t.psi_warn, t.psi_fail),
        lambda: drift.check_target_drift(train, test),
    ]

    for run_check in checks:
        try:
            report.add(run_check())
        except Exception:
            # One check failing must not take the suite down. A partial report
            # is useful; a traceback instead of a report is not.
            log.exception("A check raised and was skipped")

    return report
