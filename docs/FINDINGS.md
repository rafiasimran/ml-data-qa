# Findings

What building and running this turned up. Written for an interviewer, and for
me in six months.

---

## The check that used the wrong statistic

A missingness leak was planted deliberately: the vibration sensor drops out on
21% of defective parts and 2.9% of good ones. The fact that a reading is
*absent* carries the answer, and imputation does not remove it — the model picks
the pattern back up through the missing-value indicator.

The check was written to score the missingness indicator with AUC against the
target, with a threshold of 0.60. It passed. The leak was sitting in plain sight
and the check declared the dataset clean.

The reason is arithmetic. For a binary indicator,

```
AUC = (sensitivity + specificity) / 2
```

Here that is `(0.208 + 0.971) / 2 = 0.589`, just under the threshold. A sevenfold
difference in dropout rate — about as blatant as this defect gets — produces an
AUC barely above chance, because a binary flag can only ever move AUC so far.
Lowering the threshold would not have fixed it either: at 0.55 the check would
start firing on noise.

The fix was to change the statistic rather than the number. The check now judges
on the **ratio of missing rates between the classes**, guarded by a minimum
absolute gap so that 0.10% against 0.03% is not reported as a threefold leak.
The planted defect is now caught at 7.1×. AUC is still reported alongside,
because a reviewer will look for it, and because its weakness here is the point.

**The lesson.** A threshold that does not fire is not evidence of clean data.
I only found this because the defect had been planted on purpose and I knew what
the check was supposed to say. A validation suite that has never been tested
against a known-bad dataset is an untested test suite.

There is a regression test for this in `tests/test_leakage.py`, and it asserts
both halves: that the ratio catches the leak, and that AUC alone would not have.

---

## The random split hid the drift

The humidity sensor is recalibrated partway through the period and reads nine
points high afterwards. This is a real problem: a model fitted before the
recalibration will misread every value after it.

The drift check compared the training set against the test set and found
**PSI 0.0008** — about as stable as two samples can be.

The split was random. Pre- and post-recalibration rows are scattered evenly
across both halves, so the shift lands on each side equally and cancels. The
comparison is not wrong; it is answering a different question from the one that
matters. "Do my two halves match each other" is not "has the process changed".

Ordering the same data by time and comparing the first 70% against the last 30%
gives **PSI 1.27**.

So `DRIFT-04` was added, which reassembles train and test, sorts by timestamp
and compares early against late regardless of how the split was drawn. Production
only ever serves the latest period, so that is the comparison that predicts what
deployment will feel like.

**The lesson.** A random split does not just inflate the score. It also conceals
the drift that will degrade the model later, and it conceals it in a way that
looks like a clean bill of health.

---

## The partially cleaned model is the dangerous one

Three models, same task:

| Dataset | AUC |
|---|---|
| As delivered | 0.9325 |
| `rework_code` dropped, split untouched | 0.7007 |
| All blockers addressed, time-based split | 0.5939 |

0.93 is obviously too good — a sceptical reviewer would question it, and would
go looking for the leak.

0.70 is the problem. It is a believable number for a hard industrial
classification task. Nothing about it invites suspicion. It is also still
inflated by 286 rows sitting on both sides of the split, and by a random split
on time-ordered data.

Someone who notices the obviously wrong column, removes it, sees a plausible
score and ships it has done the visible half of the work and produced a result
that is confidently wrong.

**The lesson.** Partial cleaning can be worse than none, because it removes the
symptom that would have prompted the investigation.

---

## The honest model is barely better than chance

0.5939 AUC is a weak model. That is the correct result and it is reported as-is.

Three things hold it down, all of them real:

- the underlying signal is genuinely weak, as it often is on a production line
  where most variation is noise
- roughly 3% of labels are wrong, which puts a ceiling on any model's accuracy
  that no amount of tuning will lift
- the time-based split trains on pre-recalibration humidity and tests on
  post-recalibration humidity, so a feature the model learned to use means
  something different at test time

The useful conclusion is not "the model needs more tuning". It is that **this
data is not yet good enough to support this model**, and the next move is to fix
the sensor dropouts, the inspection consistency and the recalibration record —
not to try another algorithm.

Reporting 0.59 is more useful to a team than reporting 0.93, because 0.59 points
at what to do next.

---

## Things I would add next

- **Slice-level checks.** Overall balance can look fine while one production line
  or one shift has almost no positive examples. The model will quietly fail on
  that slice and the headline metric will not move.
- **A schema contract in version control**, so an upstream column rename fails
  loudly at ingestion rather than three checks later.
- **Confident learning for label noise.** Neighbour disagreement is cheap and
  rough; cross-validated out-of-fold probabilities would give a better estimate
  and rank individual rows for review.
- **Drift thresholds derived from history** rather than the PSI conventions of
  0.10 and 0.25, which are inherited from credit scoring and are not obviously
  right for sensor data.
