"""
The statistics the checks need, written out here instead of imported from
scikit-learn. This keeps the suite down to pandas and numpy, and makes it easy
to see exactly how each number is calculated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def auc(scores: pd.Series, labels: pd.Series) -> float:
    """
    Area under the ROC curve, via the rank identity.

    AUC equals the probability that a randomly chosen positive outranks a
    randomly chosen negative, which can be read straight off the ranks:

        AUC = (sum of positive ranks - n_pos(n_pos+1)/2) / (n_pos * n_neg)

    Ties get average ranks, which is what makes this agree with the
    trapezoidal definition rather than approximating it.
    """
    mask = scores.notna() & labels.notna()
    s, y = scores[mask], labels[mask]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = s.rank(method="average")
    pos_rank_sum = float(ranks[y == 1].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def directionless_auc(scores: pd.Series, labels: pd.Series) -> float:
    """
    AUC folded about 0.5.

    A feature with AUC 0.05 is as informative as one with 0.95, it just points
    the other way. For leak detection the direction doesn't matter, so we look
    at the distance from 0.5.
    """
    a = auc(scores, labels)
    if np.isnan(a):
        return float("nan")
    return max(a, 1.0 - a)


def target_rate_encode(values: pd.Series, labels: pd.Series) -> pd.Series:
    """
    Replace each category with the target rate observed for it.

    This turns a categorical column into something AUC can score. It overfits
    badly on high-cardinality columns, which is why the leakage check screens
    those out as ID-like first rather than reporting them as leaks.
    """
    rates = labels.groupby(values).mean()
    return values.map(rates)


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """
    Population Stability Index between two samples of one variable.

        PSI = sum over bins of (a% - e%) * ln(a% / e%)

    Bin edges come from quantiles of the reference sample, so the measure does
    not depend on the variable's units. Empty bins are floored at a small value
    so the logarithm stays finite rather than returning infinity for a single
    unobserved bucket.
    """
    e = expected.dropna()
    a = actual.dropna()
    if len(e) < bins * 5 or len(a) < bins * 5:
        return float("nan")

    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return float("nan")
    edges[0], edges[-1] = -np.inf, np.inf

    e_pct = np.histogram(e, bins=edges)[0] / len(e)
    a_pct = np.histogram(a, bins=edges)[0] / len(a)

    floor = 1e-6
    e_pct = np.clip(e_pct, floor, None)
    a_pct = np.clip(a_pct, floor, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def knn_label_disagreement(
    features: pd.DataFrame, labels: pd.Series, k: int = 10, sample: int = 1500, seed: int = 0
) -> tuple[float, int]:
    """
    Estimate label noise by asking whether each point's neighbours agree with it.

    A point whose k nearest neighbours in feature space overwhelmingly carry the
    other label is either mislabelled or genuinely unusual. Neither is provable
    from the data alone, so this reports a RATE for a human to judge, never a
    list of rows to delete automatically.

    Runs on a sample because the distance matrix is quadratic. The returned rate
    is therefore an estimate, and is reported as one.
    """
    df = features.select_dtypes(include=[np.number]).copy()
    df = df.fillna(df.median(numeric_only=True))
    if df.empty or len(df) < k + 1:
        return float("nan"), 0

    rng = np.random.default_rng(seed)
    n = min(sample, len(df))
    idx = rng.choice(len(df), size=n, replace=False)

    x = df.to_numpy(dtype=float)[idx]
    y = labels.to_numpy()[idx]

    # Standardise so a feature measured in thousands does not dominate distance.
    sd = x.std(axis=0)
    sd[sd == 0] = 1.0
    x = (x - x.mean(axis=0)) / sd

    # Squared euclidean distances via the expansion |a-b|^2 = |a|^2 - 2ab + |b|^2
    sq = (x**2).sum(axis=1)
    d = sq[:, None] - 2 * x @ x.T + sq[None, :]
    np.fill_diagonal(d, np.inf)

    neighbours = np.argpartition(d, kth=k, axis=1)[:, :k]
    neighbour_mean = y[neighbours].mean(axis=1)

    # Flag only where neighbours are lopsided: 80% agreement against the point.
    suspect = ((y == 1) & (neighbour_mean < 0.2)) | ((y == 0) & (neighbour_mean > 0.8))
    return float(suspect.mean()), int(suspect.sum())
