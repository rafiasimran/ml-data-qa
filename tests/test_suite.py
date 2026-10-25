"""The suite as a whole, and the gate it drives."""
import numpy as np

from dataqa.severity import Severity
from dataqa.suite import run_suite


def test_clean_data_has_no_blockers(clean_train, clean_test):
    report = run_suite(clean_train, clean_test, "clean")
    assert not report.failures_at(Severity.BLOCKER)
    assert report.fit_to_train


def test_one_blocker_fails_the_gate(clean_train, clean_test):
    df = clean_test.copy()
    df["rework_code"] = np.where(df.defect == 1, "RW-11", "RW-00")
    train = clean_train.copy()
    train["rework_code"] = np.where(train.defect == 1, "RW-11", "RW-00")

    report = run_suite(train, df, "leaky")
    assert not report.fit_to_train
    assert any(r.check_id == "LEAK-01" for r in report.failures_at(Severity.BLOCKER))


def test_every_check_returns_a_result(clean_train, clean_test):
    """A check that raises is swallowed so the report still renders. This asserts
    none of them are quietly doing that on ordinary input."""
    report = run_suite(clean_train, clean_test, "clean")
    assert len(report.results) == 21
    assert all(r.summary for r in report.results)


def test_failures_carry_a_remedy(clean_train, clean_test):
    train = clean_train.copy()
    train["rework_code"] = np.where(train.defect == 1, "RW-11", "RW-00")
    report = run_suite(train, clean_test, "leaky")
    for r in report.failures:
        assert r.remedy, f"{r.check_id} fails without saying what to do about it"
