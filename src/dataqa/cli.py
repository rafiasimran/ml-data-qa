"""Command line entry point: python -m dataqa check"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from .config import REPO_ROOT, thresholds
from .report import to_text, write_html
from .suite import run_suite


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dataqa", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    chk = sub.add_parser("check", help="Run the QA suite over a train/test pair")
    chk.add_argument("--train", default=str(REPO_ROOT / "data" / "train.csv"))
    chk.add_argument("--test", default=str(REPO_ROOT / "data" / "test.csv"))
    chk.add_argument("--name", default="production-line parts")
    chk.add_argument("--html", default=None, help="Also write an HTML report here")
    chk.add_argument("--json", default=None, help="Also write the raw report JSON here")
    chk.add_argument(
        "--no-gate", action="store_true",
        help="Report findings but always exit 0",
    )

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    train_path, test_path = Path(args.train), Path(args.test)
    for p in (train_path, test_path):
        if not p.exists():
            print(f"No file at {p}\nRun: python data/generate_dataset.py", file=sys.stderr)
            return 1

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    report = run_suite(train, test, dataset_name=args.name)
    print(to_text(report))

    reports_dir = thresholds.reports_dir
    json_path = Path(args.json) if args.json else reports_dir / "qa_report.json"
    report.write_json(json_path)
    html_path = Path(args.html) if args.html else reports_dir / "qa_report.html"
    write_html(report, html_path)
    print(f"\n  JSON: {json_path}\n  HTML: {html_path}")

    if args.no_gate:
        return 0
    # Exit 1 when the dataset must not be trained on, so a scheduler, a CI job
    # or a Makefile notices rather than carrying on to the training step.
    return 0 if report.fit_to_train else 1


if __name__ == "__main__":
    sys.exit(main())
