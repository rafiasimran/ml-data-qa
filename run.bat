@echo off
REM ===================================================================
REM  Run the whole project on Windows. Double-click this file, or run
REM  it from Anaconda Prompt.
REM
REM    1. generates the dataset with its planted defects
REM    2. runs the QA suite and writes a report
REM    3. trains the models that show what the defects cost
REM ===================================================================

cd /d "%~dp0"
echo.
echo === Checking Python ===
python --version
if errorlevel 1 (
  echo.
  echo Python was not found. Open "Anaconda Prompt" from the Start menu
  echo and run this file from there instead.
  pause
  exit /b 1
)

echo.
echo === Checking libraries ===
python -c "import pandas, numpy; print('pandas, numpy: OK')"
if errorlevel 1 python -m pip install pandas numpy
python -c "import sklearn" 2>nul
if errorlevel 1 (
  echo Installing scikit-learn for the experiment...
  python -m pip install scikit-learn
)

echo.
echo === Generating the dataset ===
python data\generate_dataset.py
if errorlevel 1 goto failed

echo.
echo === Running the QA suite ===
set PYTHONPATH=src
python -m dataqa check
REM Exit code 1 here is CORRECT: the dataset is deliberately broken and the
REM gate is supposed to refuse it. Not a failure of the run.

echo.
echo === What the defects cost ===
python experiments\train_model.py
if errorlevel 1 goto failed

echo.
echo ===================================================================
echo  Done.
echo    reports\qa_report.html   open this in a browser
echo    reports\qa_report.json   the same findings, machine-readable
echo ===================================================================
pause
exit /b 0

:failed
echo.
echo Something went wrong. Copy the error above.
pause
exit /b 1
