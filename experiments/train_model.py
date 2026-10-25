"""
What the data defects actually cost.

The QA suite reports problems. This script demonstrates why they matter, by
training the same model twice: once on the data exactly as delivered, and once
on data with the suite's blocking findings addressed.

The result is counter-intuitive and is the whole point of the project. The
model trained on the broken data scores far HIGHER. If the only thing you look
at is the headline metric, the defective pipeline looks like the better one.

Run:  python experiments/train_model.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dataqa.config import ID_COLUMNS, TARGET, TIME_COLUMN  # noqa: E402

LEAKY_COLUMNS = ["rework_code"]
RANDOM_STATE = 11


def _prepare(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """One-hot the categoricals; leave numerics alone. Nulls are left in place
    because HistGradientBoosting handles them natively, and imputing here would
    quietly paper over the missingness the suite flagged."""
    x = df[feature_cols].copy()
    cats = [c for c in x.columns if not pd.api.types.is_numeric_dtype(x[c])]
    return pd.get_dummies(x, columns=cats, dummy_na=True)


def _fit_and_score(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]) -> dict:
    x_train = _prepare(train, feature_cols)
    x_test = _prepare(test, feature_cols).reindex(columns=x_train.columns, fill_value=0)

    model = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.08, random_state=RANDOM_STATE
    )
    model.fit(x_train, train[TARGET])
    proba = model.predict_proba(x_test)[:, 1]

    return {
        "auc": round(float(roc_auc_score(test[TARGET], proba)), 4),
        "average_precision": round(float(average_precision_score(test[TARGET], proba)), 4),
        "train_rows": len(train),
        "test_rows": len(test),
        "n_features": x_train.shape[1],
    }


def run_as_delivered(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Everything the export contained, split exactly as it arrived."""
    feature_cols = [
        c for c in train.columns if c not in {TARGET, TIME_COLUMN, *ID_COLUMNS}
    ]
    return _fit_and_score(train, test, feature_cols)


def run_qa_cleaned(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """
    The same task with the suite's blocking findings addressed.

    Four changes, each tied to a specific check:
      LEAK-01   drop rework_code, which is written after the label exists
      LEAK-02   drop vibration_mm_s, whose absence encodes the outcome
      SPLIT-01  deduplicate, then rebuild the split so no row is on both sides
      SPLIT-02  split on time, so the model is tested on a period it has not seen
    """
    combined = pd.concat([train, test], ignore_index=True)
    combined = combined.drop_duplicates(subset=ID_COLUMNS, keep="first")
    combined["_ts"] = pd.to_datetime(combined[TIME_COLUMN], errors="coerce")
    combined = combined.dropna(subset=["_ts"]).sort_values("_ts").reset_index(drop=True)

    cut = int(len(combined) * 0.75)
    tr, te = combined.iloc[:cut].copy(), combined.iloc[cut:].copy()

    drop = {TARGET, TIME_COLUMN, "_ts", *ID_COLUMNS, *LEAKY_COLUMNS, "vibration_mm_s"}
    feature_cols = [c for c in tr.columns if c not in drop]
    return _fit_and_score(tr, te, feature_cols)


def run_partially_cleaned(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """
    The tempting middle option: drop the obvious leak, keep the broken split.

    Worth measuring because it is what most people do. Removing the column that
    looks wrong feels like fixing the problem, while the contaminated split goes
    unnoticed and keeps inflating the score.
    """
    feature_cols = [
        c for c in train.columns
        if c not in {TARGET, TIME_COLUMN, *ID_COLUMNS, *LEAKY_COLUMNS}
    ]
    return _fit_and_score(train, test, feature_cols)


def main() -> int:
    data = REPO_ROOT / "data"
    if not (data / "train.csv").exists():
        print("No data. Run: python data/generate_dataset.py", file=sys.stderr)
        return 1

    train = pd.read_csv(data / "train.csv")
    test = pd.read_csv(data / "test.csv")

    print("Training three models. This takes under a minute.\n")
    results = {
        "as_delivered": run_as_delivered(train, test),
        "leak_column_removed_only": run_partially_cleaned(train, test),
        "qa_cleaned": run_qa_cleaned(train, test),
    }

    width = 78
    print("=" * width)
    print("  WHAT THE DATA DEFECTS COST")
    print("=" * width)
    rows = [
        ("As delivered", "as_delivered",
         "everything in the export, split as it arrived"),
        ("Leak column removed", "leak_column_removed_only",
         "rework_code dropped, split left alone"),
        ("QA-cleaned", "qa_cleaned",
         "blockers addressed, deduplicated, time-based split"),
    ]
    print(f"\n  {'':<22}{'AUC':>8}{'Avg precision':>16}   {'':<4}")
    print("  " + "-" * (width - 4))
    for label, key, note in rows:
        r = results[key]
        print(f"  {label:<22}{r['auc']:>8.4f}{r['average_precision']:>16.4f}")
        print(f"  {'':<22}{note}")
    print("  " + "-" * (width - 4))

    delta = results["as_delivered"]["auc"] - results["qa_cleaned"]["auc"]
    print(f"""
  The model trained on the data as delivered scores {delta:.3f} AUC HIGHER than
  the one trained on cleaned data. That gap is not skill. It is the model
  reading a column filled in after the outcome was known, and being tested
  on rows it had already memorised.

  The honest number is {results['qa_cleaned']['auc']:.4f}. It is the lower one,
  and it is the only one that would survive contact with production.
""")
    print("=" * width)

    out = REPO_ROOT / "reports" / "experiment_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  Written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
