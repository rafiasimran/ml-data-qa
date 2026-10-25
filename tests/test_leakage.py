"""
The leakage checks, which are the ones that have to work.

Each test plants one leak in otherwise clean data and asserts the check finds it,
then asserts the same check stays quiet on clean data. A check that never passes
is as useless as one that never fails.
"""
import numpy as np
import pandas as pd

from dataqa.checks.leakage import (
    check_id_like_columns,
    check_missingness_leakage,
    check_single_feature_leakage,
)


def test_catches_a_column_written_after_the_outcome(clean_train):
    df = clean_train.copy()
    # A code the inspector assigns once the verdict is known.
    df["rework_code"] = np.where(df.defect == 1, "RW-11", "RW-00")
    result = check_single_feature_leakage(df, auc_threshold=0.80)
    assert not result.passed
    assert "rework_code" in result.evidence["suspects"]


def test_quiet_on_clean_data(clean_train):
    assert check_single_feature_leakage(clean_train, auc_threshold=0.80).passed


def test_does_not_mistake_an_identifier_for_a_leak(clean_train):
    """part_id is unique per row, so target-rate encoding it scores a perfect
    AUC that means nothing. It must be screened out, not reported as a leak."""
    df = clean_train.copy()
    df["serial_no"] = [f"SN-{i}" for i in range(len(df))]
    result = check_single_feature_leakage(df, auc_threshold=0.80)
    assert "serial_no" in result.evidence["id_like_columns_skipped"]
    assert "serial_no" not in result.evidence["suspects"]


def test_catches_missingness_that_encodes_the_target(clean_train):
    """The regression test for the real bug this check had: a sensor dropping
    out far more often on positives scores only ~0.59 AUC, which slipped under
    an AUC threshold. The check now judges on the ratio of missing rates."""
    df = clean_train.copy()
    rng = np.random.default_rng(3)
    drop = (df.defect == 1) & (rng.random(len(df)) < 0.25)
    df.loc[drop, "vibration_mm_s"] = np.nan
    # A little missingness on the other class too, so the ratio is finite.
    df.loc[(df.defect == 0) & (rng.random(len(df)) < 0.03), "vibration_mm_s"] = np.nan

    result = check_missingness_leakage(df, auc_threshold=0.60)
    assert not result.passed
    assert "vibration_mm_s" in result.evidence["suspects"]

    # Why the old AUC version missed it: the sensor drops out several times
    # more often on one class, but the indicator's AUC stays close to 0.5
    # because a yes/no flag's AUC is (sensitivity + specificity) / 2.
    # Not pinning an exact AUC, so the test isn't brittle.
    found = result.evidence["by_column"]["vibration_mm_s"]
    assert found["rate_ratio"] > 5.0
    assert found["auc_of_indicator"] < 0.65


def test_ignores_missingness_unrelated_to_the_target(clean_train):
    df = clean_train.copy()
    rng = np.random.default_rng(4)
    df.loc[rng.random(len(df)) < 0.12, "humidity_pct"] = np.nan
    assert check_missingness_leakage(df, auc_threshold=0.60).passed


def test_ignores_a_tiny_missingness_gap(clean_train):
    """0.2% against 0.05% is a fourfold ratio and still noise. The absolute-gap
    guard is what stops the check crying wolf."""
    df = clean_train.copy()
    df.loc[df.index[:1], "temp_c"] = np.nan
    assert check_missingness_leakage(df, auc_threshold=0.60).passed


def test_flags_identifier_like_feature_columns(clean_train):
    df = clean_train.copy()
    df["batch_ref"] = [f"B-{i}" for i in range(len(df))]
    result = check_id_like_columns(df)
    assert not result.passed
    assert "batch_ref" in result.evidence["distinct_value_ratio"]
