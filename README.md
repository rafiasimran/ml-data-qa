# Data QA for Machine Learning

[![Tests](https://github.com/rafiasimran/ml-data-qa/actions/workflows/tests.yml/badge.svg)](https://github.com/rafiasimran/ml-data-qa/actions/workflows/tests.yml)

A test suite for the data that machine learning models are trained and tested on.

In software we test code before it ships, but training data often goes straight into a model without anyone checking it. I come from software QA, so I wanted to try applying the same idea to data: 21 checks, each with a severity, a short finding and a suggested fix, and a gate that stops a broken dataset from going to training.

```
python data/generate_dataset.py     # makes a dataset with known problems in it
python -m dataqa check              # runs the checks, exits with 1 if the data isn't fit to train
python experiments/train_model.py   # shows what those problems do to a model
```

**On Windows:** double-click `run.bat`, or see [docs/WINDOWS.md](docs/WINDOWS.md).

An example report is in [`docs/sample_report.html`](docs/sample_report.html). GitHub only shows the source, so download it and open it in a browser to see it properly.

## Main result

I trained the same model on the same task three times:

| Dataset | AUC | Average precision |
|---|---|---|
| As delivered | **0.9325** | 0.8815 |
| Obvious leak removed, split unchanged | 0.7007 | 0.4268 |
| All QA findings fixed | **0.5939** | 0.2388 |

The broken data gives a 0.34 higher AUC. That's not because the model is better. It's using a column that was filled in after the result was known, and it's being tested on rows it already saw in training.

So the real number is the lowest one. If you only looked at the first score you would think the model works well, and then it would fail once it's used on new data.

The middle row surprised me most. Removing the column that is obviously wrong feels like it fixes things, but the score is still too high because of the overlapping split. 0.70 looks believable, so nobody would question it.

## What it checks

21 checks in five groups.

**Leakage**, information in the features that won't exist when you actually make a prediction. These are all BLOCKERs because leakage makes the score go up, not down.

- `LEAK-01` no single feature predicts the target almost on its own
- `LEAK-02` whether a value is missing doesn't predict the target
- `LEAK-03` no ID-like columns in the features

**Split integrity**, whether the test score can be trusted at all.

- `SPLIT-01` no rows in both train and test (checked by ID and by row content)
- `SPLIT-02` the split follows time order
- `DUP-01` / `DUP-02` no duplicate rows or repeated IDs

**Target**: `LABEL-01` to `LABEL-04` check that the label is there and complete, only has the expected values, the classes aren't too unbalanced, and give a rough estimate of label noise.

**Drift**: `DRIFT-01` to `DRIFT-04` compare feature distributions (PSI), categories, the target rate, and how features change over time no matter how the split was made.

**Structure**: `SCHEMA-01` to `SCHEMA-04`, `RANGE-01`, `NULL-01` check columns and types, that train and test match, constant columns, impossible values and missing values.

## Severity levels

```
BLOCKER   the model will be wrong in a way the metrics hide
MAJOR     the model will be worse, or a metric will be misleading
MINOR     worth fixing, but little effect on the model
INFO      just describes the dataset
```

One BLOCKER fails the gate and the command exits with code 1. I didn't add an overall percentage score on purpose. A dataset with leakage isn't "94% fine", it can't be used.

## Does it actually catch the problems?

The generator puts eight kinds of problems into the data in known amounts and writes them to a manifest file. Then I ran the suite to check each one gets found:

| Planted problem | Caught by | What the suite reported |
|---|---|---|
| Column filled in after the label | `LEAK-01` | `rework_code`, 0.925 AUC on its own |
| Sensor missing more often for one class | `LEAK-02` | missing 7.1x more often for defects |
| 180 rows copied into both splits | `SPLIT-01` | 286 rows on both sides |
| Random split on time-ordered data | `SPLIT-02` | 100% of test rows older than the end of train |
| 300 duplicated rows | `DUP-01`, `DUP-02` | 47 in train, 170 repeated IDs |
| 140 impossible sensor values | `RANGE-01` | 104 out of range in train |
| Sensor recalibrated in the middle | `DRIFT-04` | `humidity_pct`, PSI 1.27 |
| A constant column | `SCHEMA-04` | `plant_code` |
| 3% of labels flipped | `LABEL-04` | about 6.9% estimated |

The last one overestimates, because an unusual but correct part looks the same as a wrong label. So the check only reports a rate and doesn't delete anything.

## Design choices

**Leakage blocks the gate instead of giving a warning.** A leak means the model is invalid, so it shouldn't be averaged into a health score.

**The missingness check doesn't use AUC.** My first version did, and it missed a leak I had planted on purpose. For a yes/no flag, AUC = (sensitivity + specificity) / 2, so a sensor missing on 21% of bad parts and 3% of good parts only scores 0.59. Now the check compares the missing rates directly. More on this in [docs/FINDINGS.md](docs/FINDINGS.md).

**Drift is also checked over time.** With a random split, train and test always look the same, because a change in the middle of the period ends up in both halves. Here the recalibrated sensor has PSI 0.0008 between train and test, but PSI 1.27 when I compare early data with late data.

## Project structure

```
src/dataqa/
  severity.py    severity levels, results, the gate
  config.py      thresholds and why I chose them
  stats.py       AUC, PSI and neighbour disagreement, written by hand
  checks/        one file per group of checks
  suite.py       runs all checks and builds the report
  report.py      terminal and HTML output
  cli.py         python -m dataqa check
data/            dataset generator
experiments/     model comparison (broken vs cleaned data)
tests/           25 tests
docs/            findings, Windows guide, sample report
```

The checks only need pandas and numpy. scikit-learn is only used for the experiment.

## Tech

Python, pandas, numpy, scikit-learn (experiment only), pytest, ruff, GitHub Actions

## Licence

MIT, see [LICENSE](LICENSE).
