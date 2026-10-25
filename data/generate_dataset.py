"""
Generate a synthetic production-line dataset with deliberately planted defects.

Every defect below is one that really occurs in industrial ML datasets and that
really damages a model. Each is planted in a known quantity and written to a
manifest, so the QA suite can be proven to catch it rather than assumed to.

Scenario: parts come off a production line. Sensors record the conditions each
part was made under. Quality control later marks the part pass or fail. The
modelling task is to predict the failure before QC sees it.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

LINES = ["LINE-A", "LINE-B", "LINE-C"]
SHIFTS = ["early", "late", "night"]
TOOLS = [f"T-{i:02d}" for i in range(1, 9)]


def _physics(rng: random.Random, tool_wear: float, shift: str) -> dict:
    """Sensor readings, loosely coupled so the task is learnable but not trivial."""
    temp = rng.gauss(212 + tool_wear * 0.04, 6)
    pressure = rng.gauss(88 - tool_wear * 0.01, 3)
    vibration = rng.gauss(0.42 + tool_wear * 0.0009, 0.08)
    cycle_time = rng.gauss(31 + (2.5 if shift == "night" else 0), 2.2)
    humidity = rng.gauss(44, 7)
    return {
        "temp_c": round(temp, 2),
        "pressure_bar": round(pressure, 2),
        "vibration_mm_s": round(vibration, 4),
        "cycle_time_s": round(cycle_time, 2),
        "humidity_pct": round(humidity, 1),
    }


def _defect_probability(row: dict, tool_wear: float) -> float:
    """
    The true data-generating process.

    Kept weak on purpose. Real production data is mostly noise, so a very
    high AUC here would be unrealistic.
    """
    z = (
        -3.05
        + 0.020 * (row["temp_c"] - 212)
        + 0.055 * (88 - row["pressure_bar"])
        + 3.10 * (row["vibration_mm_s"] - 0.42)
        + 0.0021 * tool_wear
        + 0.012 * (row["humidity_pct"] - 44)
    )
    return 1 / (1 + math.exp(-z))


def generate(n_rows: int, seed: int) -> tuple[pd.DataFrame, dict]:
    rng = random.Random(seed)
    manifest: dict = {"n_rows_requested": n_rows, "seed": seed, "planted": {}}

    start = datetime(2026, 1, 6, 6, 0, 0)
    rows: list[dict] = []

    for i in range(n_rows):
        # Parts are produced in time order. This matters later: a random split
        # on time-ordered data lets the model see the future.
        ts = start + timedelta(minutes=4 * i)
        shift = SHIFTS[(ts.hour // 8) % 3]
        tool = rng.choice(TOOLS)

        # Tool wear climbs through a shift and resets on tool change.
        tool_wear = (i % 420) * 0.9

        row = {
            "part_id": f"P{i:07d}",
            "produced_at": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "line": rng.choice(LINES),
            "shift": shift,
            "tool_id": tool,
            "tool_wear_min": round(tool_wear, 1),
        }
        row.update(_physics(rng, tool_wear, shift))

        p = _defect_probability(row, tool_wear)
        row["defect"] = 1 if rng.random() < p else 0

        # --- PLANTED DEFECT 1: target leakage -------------------------------
        # rework_code is assigned BY the QC inspector, after the label exists.
        # It is in the export because it lives in the same QC table. A model
        # given this column scores beautifully and is worthless in production,
        # because at prediction time the value does not exist yet.
        row["rework_code"] = (
            rng.choice(["RW-11", "RW-12", "RW-20"]) if row["defect"] == 1
            else ("RW-00" if rng.random() > 0.02 else "RW-11")
        )

        # --- PLANTED DEFECT 2: a constant column ----------------------------
        row["plant_code"] = "MUC-1"

        # --- PLANTED DEFECT 3: missingness that encodes the target ----------
        # The vibration sensor drops out more often on bad parts, so the fact
        # that the value is MISSING is itself a clue. Imputing the mean hides
        # the leak without removing it.
        if row["defect"] == 1 and rng.random() < 0.22:
            row["vibration_mm_s"] = None
        elif rng.random() < 0.03:
            row["vibration_mm_s"] = None

        rows.append(row)

    df = pd.DataFrame(rows)
    manifest["planted"]["target_leakage_columns"] = ["rework_code"]
    manifest["planted"]["constant_columns"] = ["plant_code"]
    manifest["planted"]["missingness_leak_column"] = "vibration_mm_s"

    # --- PLANTED DEFECT 4: label noise --------------------------------------
    # QC inspectors disagree. Flip a known share of labels.
    n_flip = int(len(df) * 0.03)
    flip_idx = rng.sample(range(len(df)), n_flip)
    df.loc[flip_idx, "defect"] = 1 - df.loc[flip_idx, "defect"]
    manifest["planted"]["labels_flipped"] = n_flip

    # --- PLANTED DEFECT 5: exact duplicate rows -----------------------------
    # The MES exported one batch twice. Same part_id, same everything.
    n_dupes = int(len(df) * 0.025)
    dupe_rows = df.sample(n=n_dupes, random_state=seed)
    df = pd.concat([df, dupe_rows], ignore_index=True)
    manifest["planted"]["duplicate_rows"] = n_dupes

    # --- PLANTED DEFECT 6: physically impossible values ---------------------
    n_impossible = 140
    imp_idx = rng.sample(range(len(df)), n_impossible)
    for j, idx in enumerate(imp_idx):
        if j % 2 == 0:
            df.at[idx, "pressure_bar"] = round(-abs(rng.gauss(40, 10)), 2)
        else:
            df.at[idx, "temp_c"] = 9999.0
    manifest["planted"]["impossible_values"] = n_impossible

    # --- PLANTED DEFECT 7: sensor recalibration partway through -------------
    # The humidity sensor was recalibrated and reads 9 points high afterwards.
    # Train on the early period, test on the late one, and the feature drifts.
    cutoff = int(len(df) * 0.70)
    df.loc[cutoff:, "humidity_pct"] = df.loc[cutoff:, "humidity_pct"] + 9.0
    manifest["planted"]["drifted_column"] = "humidity_pct"
    manifest["planted"]["drift_starts_at_row"] = cutoff

    # Shuffle, because a real export is not sorted helpfully for you.
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    manifest["n_rows_final"] = len(df)
    manifest["defect_rate_pct"] = round(100 * df["defect"].mean(), 2)
    return df, manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    df, manifest = generate(args.rows, args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- PLANTED DEFECT 8: train/test overlap -------------------------------
    # The split was made by sampling rows at random AFTER the duplicates were
    # introduced, so the same part appears on both sides. The test set is no
    # longer unseen, and every score computed on it is optimistic.
    shuffled = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    cut = int(len(shuffled) * 0.75)
    train, test = shuffled.iloc[:cut].copy(), shuffled.iloc[cut:].copy()

    overlap = train.sample(n=180, random_state=args.seed)
    test = pd.concat([test, overlap], ignore_index=True)
    manifest["planted"]["train_test_overlap_rows"] = 180

    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    (out / "planted_defects.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"train.csv : {len(train):,} rows")
    print(f"test.csv  : {len(test):,} rows")
    print(f"defect rate: {manifest['defect_rate_pct']}%")
    print(f"manifest  : {out / 'planted_defects.json'}")


if __name__ == "__main__":
    main()
