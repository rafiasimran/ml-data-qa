# Findings

Notes on what I found while building and running this.

## 1. The missingness check used the wrong statistic

I planted a leak on purpose: the vibration sensor is missing for 21% of defective parts but only 2.9% of good ones. So the fact that a value is missing gives away the answer. Imputing doesn't fix it, because the model can still pick up the pattern from the missing-value indicator.

My first version scored the missing-value indicator with AUC and used a threshold of 0.60. It passed, even though the leak was clearly there.

The reason is that for a yes/no indicator:

```
AUC = (sensitivity + specificity) / 2
```

Here that's `(0.208 + 0.971) / 2 = 0.589`, just below 0.60. A 7x difference in how often the sensor drops out only gives an AUC a little above chance. Lowering the threshold wouldn't really help either, at 0.55 it would start flagging noise.

So I changed the statistic instead of the threshold. The check now compares the missing rates of the two classes (ratio), with a minimum gap so that something like 0.10% vs 0.03% isn't flagged. The planted leak now shows up at 7.1x. I kept the AUC in the output so you can see the difference.

What I learned: a check that doesn't fire doesn't mean the data is clean. I only noticed because I knew the leak was there. There's a regression test for this in `tests/test_leakage.py` that checks both things: the ratio catches it, and AUC alone wouldn't have.

## 2. The random split hid the drift

In the generated data the humidity sensor is recalibrated partway through and reads about nine points higher after that. A model trained on the earlier data would misread everything after it.

The normal drift check (train vs test) gave **PSI 0.0008**, so basically no drift.

That's because the split was random. Rows from before and after the recalibration are spread evenly across both halves, so the difference cancels out. When I sorted the same data by time and compared the first 70% with the last 30%, PSI was **1.27**.

So I added `DRIFT-04`, which does exactly that, regardless of how the split was made. In production the model only sees new data, so this is the comparison that matters.

## 3. Partly cleaned data is the risky case

Three models on the same task:

| Dataset | AUC |
|---|---|
| As delivered | 0.9325 |
| `rework_code` dropped, split unchanged | 0.7007 |
| All blockers fixed, split by time | 0.5939 |

0.93 is obviously too good, so someone would probably question it.

0.70 is the problem. It looks like a normal result for a hard industrial task, so nobody would look closer. But it's still too high because 286 rows are in both train and test, and the split is random on time-ordered data.

If you remove the column that looks wrong and stop there, you end up with a result that looks fine and is still wrong.

## 4. The honest model is weak

0.5939 AUC is not a good model, but it's the correct result for this data, so I'm reporting it as it is.

Three reasons it's low:

- the real signal is weak, which is normal for production data where most variation is noise
- about 3% of labels are wrong, which limits how good any model can get
- with the time-based split, the model trains on humidity before the recalibration and is tested on humidity after it

So the next step isn't more tuning or a different algorithm. It's fixing the data: the sensor dropouts, the inspection labels and the recalibration.

## What I'd add next

- **Checks per slice.** The overall class balance can look fine while one production line or shift has almost no defects.
- **A schema file in version control**, so a renamed column fails right away.
- **A better label noise method.** The neighbour method is quick but rough. Out-of-fold predictions from cross-validation would give a better estimate and could rank rows for review.
- **Drift thresholds based on past data.** The 0.10 and 0.25 PSI limits come from credit scoring and might not be right for sensor data.
