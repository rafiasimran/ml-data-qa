# Running this on Windows

## The easy way

Double-click **`run.bat`**. It checks Python, installs anything missing,
generates the dataset, runs the QA suite and then trains the comparison models.

If double-clicking does nothing, open **Anaconda Prompt** from the Start menu,
`cd` to this folder, and run `run.bat` from there.

Note that the QA step exits with code 1. That's expected, the dataset is
broken on purpose and the gate should refuse it.

## PowerShell, step by step

```powershell
cd path\to\ml-data-qa

python -m pip install pandas numpy scikit-learn

python data\generate_dataset.py

$env:PYTHONPATH = "src"
python -m dataqa check

python experiments\train_model.py
```

`$env:PYTHONPATH = "src"` goes on its own line. The `PYTHONPATH=src python ...`
form used in the README is Unix syntax and will not work here.

### Command Prompt

```cmd
set PYTHONPATH=src
python -m dataqa check
```

## Reading the output

- `reports\qa_report.html`: opens in a browser. Failures are listed first,
  worst first, and each one has an evidence section you can expand.
- `reports\qa_report.json`: the same findings as JSON, e.g. for CI.
- `reports\experiment_results.json`: the scores of the three models.

## Things to try

**Watch the gate pass.** Point the suite at a dataset with the blockers removed
and the gate goes green:

```powershell
python -c "import pandas as pd; d=pd.read_csv('data/train.csv').drop(columns=['rework_code']); d.to_csv('data/train_clean.csv', index=False)"
```

**Loosen the leakage threshold** and watch `rework_code` slip through, to see
why the threshold matters:

```powershell
$env:LEAKAGE_AUC = "0.95"
python -m dataqa check
Remove-Item Env:\LEAKAGE_AUC
```

**Run the tests.**

```powershell
python -m pip install pytest
python -m pytest -v
```

## If something breaks

**`'python' is not recognized`**: use Anaconda Prompt, or add Python to PATH.

**`ModuleNotFoundError: No module named 'dataqa'`**: `PYTHONPATH` isn't set.
Run `$env:PYTHONPATH = "src"` from inside the project folder.

**`No file at ...train.csv`**: run `python data\generate_dataset.py` first.

**The experiment is slow**: training three models on 12,000 rows should take
less than a minute. If it hangs, check scikit-learn installed
cleanly.
