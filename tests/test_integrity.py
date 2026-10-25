"""Split integrity — the checks that decide whether a score can be believed."""
import pandas as pd

from dataqa.checks.integrity import (
    check_duplicate_ids,
    check_duplicate_rows,
    check_temporal_split,
    check_train_test_overlap,
)


def test_clean_split_has_no_overlap(clean_train, clean_test):
    assert check_train_test_overlap(clean_train, clean_test, max_overlap=0).passed


def test_catches_rows_present_on_both_sides(clean_train, clean_test):
    contaminated = pd.concat([clean_test, clean_train.head(20)], ignore_index=True)
    result = check_train_test_overlap(clean_train, contaminated, max_overlap=0)
    assert not result.passed
    assert result.evidence["shared_id_values"] == 20


def test_catches_overlap_hidden_by_a_fresh_identifier(clean_train, clean_test):
    """Copied rows given new IDs still look identical on content, which is how
    an overlap survives a check that only compares identifiers."""
    smuggled = clean_train.head(15).copy()
    smuggled["part_id"] = [f"Z{i:05d}" for i in range(15)]
    contaminated = pd.concat([clean_test, smuggled], ignore_index=True)

    result = check_train_test_overlap(clean_train, contaminated, max_overlap=0)
    assert not result.passed
    assert result.evidence["shared_id_values"] == 0
    assert result.evidence["identical_content_rows"] == 15


def test_overlap_check_survives_nulls(clean_train, clean_test):
    """Row hashing has to cope with NaN beside text; an earlier version raised
    TypeError here and took the whole check out of the report."""
    train = clean_train.copy()
    train.loc[train.index[:30], "vibration_mm_s"] = None
    test = clean_test.copy()
    test.loc[test.index[:10], "humidity_pct"] = None
    result = check_train_test_overlap(train, test, max_overlap=0)
    assert result.passed


def test_catches_duplicate_rows(clean_train):
    df = pd.concat([clean_train, clean_train.head(40)], ignore_index=True)
    assert not check_duplicate_rows(df, max_pct=0.5).passed
    assert not check_duplicate_ids(df).passed


def test_random_split_is_flagged_as_not_chronological(clean_train):
    """A random split leaves test rows scattered through the training period."""
    shuffled = clean_train.sample(frac=1.0, random_state=0).reset_index(drop=True)
    cut = int(len(shuffled) * 0.75)
    result = check_temporal_split(shuffled.iloc[:cut], shuffled.iloc[cut:])
    assert not result.passed


def test_chronological_split_passes(clean_train):
    ordered = clean_train.sort_values("produced_at").reset_index(drop=True)
    cut = int(len(ordered) * 0.75)
    assert check_temporal_split(ordered.iloc[:cut], ordered.iloc[cut:]).passed
