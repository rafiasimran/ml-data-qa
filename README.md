# Data QA for Machine Learning

[![Tests](https://github.com/rafiasimran/ml-data-qa/actions/workflows/tests.yml/badge.svg)](https://github.com/rafiasimran/ml-data-qa/actions/workflows/tests.yml)

A test suite for the datasets that train and evaluate models.

Software gets tested before it ships. Training data usually does not, which is
why so many models look excellent offline and disappoint in production. This
project applies the same discipline to data: twenty-one test cases, each with a
severity, a plain-language finding and a remedy, behind a gate that refuses to
let a broken dataset through to training.

```
python data/generate_dataset.py     # a dataset with known, deliberate defects
python -m dataqa check              # the suite — exits 1 if not fit to train
python experiments/train_model.py   # what those defects cost
```

**On Windows:** double-click `run.bat`. See [docs/WINDOWS.md](docs/WINDOWS.md).

A sample of the report it produces is committed at
[`docs/sample_report.html`](docs/sample_report.html) — GitHub shows the source,
so to see it rendered either download it, or open it through
`htmlpreview.github.io/?` followed by the file's URL.

---

## The result that makes the point

The same model, the same task, trained three times on the same source data.

| Dataset | AUC | Average precision |
|---|---|---|
| As delivered | **0.9325** | 0.8815 |
| Obvious leak removed, split untouched | 0.7007 | 0.4268 |
| QA findings addressed | **0.5939** | 0.2388 |

The broken dataset scores **0.34 AUC higher**. That gap is not skill. It is the
model reading a column filled in after the outcome was known, and being scored
on rows it had already memorised.

The honest number is the lowest one. A team shipping the first model would have
reported 0.93 to their stakeholders and watched it collapse on deployment, with
nothing in the offline metrics to warn them.

The middle row is the one worth dwelling on. Dropping the column that obviously
looks wrong feels like fixing the problem, and still leaves a score inflated by
a contaminated split. Partial cleaning produces a number that is wrong and
plausible, which is more dangerous than one that is wrong and obvious.

---

## What it checks

Twenty-one checks in five families. Each returns a severity, a finding and a
remedy.

**Leakage** — information in the features that will not exist at prediction time.
Every check here is a BLOCKER, because leakage makes a model look *better* the
more broken it is.

- `LEAK-01` no single feature predicts the target almost alone
- `LEAK-02` the missingness pattern does not predict the target
- `LEAK-03` no identifier-like columns sitting among the features

**Split integrity** — whether the reported score means anything at all.

- `SPLIT-01` train and test do not overlap *(by identifier and by row content)*
- `SPLIT-02` the split respects time order rather than cutting through it
- `DUP-01` / `DUP-02` no duplicate rows, no repeated identifiers

**Target** — `LABEL-01` to `LABEL-04`: present and complete, only expected values,
usably balanced, and an estimate of label noise from neighbour disagreement.

**Drift** — `DRIFT-01` to `DRIFT-04`: feature distributions by PSI, categorical
levels consistent across splits, target rate stable, and features stable *over
time* independently of how the split was drawn.

**Structure** — `SCHEMA-01` to `SCHEMA-04`, `RANGE-01`, `NULL-01`: expected columns
and types, train and test agree, no constant columns, values physically possible,
missing rates within limits.

---

## Severity, and why the gate is not a score

```
BLOCKER   the model will be wrong in a way the metrics hide
MAJOR     the model will be measurably worse, or a metric will mislead
MINOR     worth fixing, no real effect on the model
INFO      recorded to describe the dataset, not a fault
```

One BLOCKER fails the gate and the CLI exits non-zero. There is no percentage
score, deliberately: a dataset with leakage is not 94% fine. It is unusable, and
averaging that away into a number is how a quality gate becomes decoration.

---

## Proving the suite works

The dataset generator plants eight classes of defect in known quantities and
writes a manifest. The suite is then run against it, and every planted defect is
accounted for.

| Planted | Caught by | What the suite reported |
|---|---|---|
| Column written after the label exists | `LEAK-01` | `rework_code`, 0.925 AUC alone |
| Sensor dropping out more on one class | `LEAK-02` | missing 7.1× more often on defects |
| 180 rows copied into both splits | `SPLIT-01` | 286 rows on both sides |
| Random split on time-ordered data | `SPLIT-02` | 100% of test rows predate training |
| 300 duplicated rows | `DUP-01`, `DUP-02` | 47 in train, 170 repeated IDs |
| 140 impossible sensor values | `RANGE-01` | 104 out of range in train |
| Sensor recalibrated mid-period | `DRIFT-04` | `humidity_pct`, PSI 1.27 |
| A constant column | `SCHEMA-04` | `plant_code` |
| 3% of labels flipped | `LABEL-04` | ~6.9% estimated by neighbour disagreement |

The last row is honest rather than flattering: the estimate overshoots, because
a genuinely unusual part looks the same as a mislabelled one. The check reports
a rate for a person to judge and never deletes a row on its own.

---

## Three things worth defending

**Leakage is a BLOCKER, not a warning.** Most data quality tooling scores
datasets and lets the user decide. A leak is not a matter of degree — the model
is invalid and the metric is actively lying about it. Anything that can be
averaged into an overall health score will be.

**The missingness check does not use AUC.** It judges on the ratio of missing
rates between classes. AUC is the wrong instrument for a binary indicator: it is
bounded by (sensitivity + specificity) / 2, so a sensor dropping out on 21% of
bad parts and 3% of good ones — a sevenfold difference and an obvious leak —
scores 0.59 and slips under any sane threshold. This was found by running the
suite against a leak that had been planted deliberately and watching it pass.
See [docs/FINDINGS.md](docs/FINDINGS.md).

**Drift is measured over time, not just across the split.** Comparing train
against test only shows whether the two halves match. When the split is random
they always will, because a mid-period process change lands on both sides
equally and cancels out. In this dataset the recalibrated sensor shows PSI
0.0008 between train and test — perfectly stable — and PSI 1.27 when the same
data is compared early period against late. Production only ever sees the late
period.

---

## Layout

```
src/dataqa/
  severity.py    Severity, CheckResult, QAReport, the gate
  config.py      thresholds, each with its reasoning
  stats.py       AUC, PSI, neighbour disagreement — written out, not imported
  checks/        one module per defect family
  suite.py       runs everything, assembles the report
  report.py      terminal and HTML output
  cli.py         python -m dataqa check
data/            generator, plus the manifest of what it planted
experiments/     the dirty-versus-clean model comparison
tests/           25 tests — every check, against data broken on purpose
docs/            findings, Windows quickstart
```

The suite depends on pandas and numpy alone, so it runs wherever the data is.
scikit-learn is needed only by the experiment.

## Tech

Python · pandas · numpy · scikit-learn (experiment only) · pytest · ruff

## Licence

MIT — see [LICENSE](LICENSE).
