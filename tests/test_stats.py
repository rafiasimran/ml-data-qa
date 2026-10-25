"""The statistics, checked against cases with known answers."""
import numpy as np
import pandas as pd

from dataqa.stats import auc, directionless_auc, psi


def test_auc_of_a_perfect_ranking_is_one():
    y = pd.Series([0, 0, 0, 1, 1, 1])
    assert auc(pd.Series([1, 2, 3, 4, 5, 6]), y) == 1.0


def test_auc_of_a_reversed_ranking_is_zero_and_folds_to_one():
    y = pd.Series([0, 0, 0, 1, 1, 1])
    s = pd.Series([6, 5, 4, 3, 2, 1])
    assert auc(s, y) == 0.0
    assert directionless_auc(s, y) == 1.0


def test_auc_of_a_constant_score_is_one_half():
    y = pd.Series([0, 0, 1, 1])
    assert auc(pd.Series([5, 5, 5, 5]), y) == 0.5


def test_auc_is_nan_when_a_class_is_absent():
    assert np.isnan(auc(pd.Series([1, 2, 3]), pd.Series([1, 1, 1])))


def test_psi_is_near_zero_for_the_same_distribution():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 1, 5000))
    b = pd.Series(rng.normal(0, 1, 5000))
    assert psi(a, b) < 0.05


def test_psi_is_large_for_a_shifted_distribution():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 1, 5000))
    b = pd.Series(rng.normal(1.5, 1, 5000))
    assert psi(a, b) > 0.25


def test_psi_is_nan_on_samples_too_small_to_bin():
    assert np.isnan(psi(pd.Series([1, 2, 3]), pd.Series([1, 2, 3])))
