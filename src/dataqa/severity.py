"""
Result types for the suite.

Modelled on a software QA test report rather than a statistics summary, because
the job is the same one: decide whether this artefact is fit to ship. A dataset
that fails a BLOCKER check should not be trained on, in the same way a build
that fails a smoke test should not be released.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


class Severity(str, Enum):
    """
    Severity is about consequence, not about how unusual the finding is.

    BLOCKER  the model trained on this will be wrong in a way the metrics hide.
             Leakage is the archetype: the score goes UP and the model is junk.
    MAJOR    the model will be measurably worse, or a metric will mislead.
    MINOR    worth fixing, no real effect on the model.
    INFO     recorded so the dataset can be described, not a fault.
    """

    BLOCKER = "BLOCKER"
    MAJOR = "MAJOR"
    MINOR = "MINOR"
    INFO = "INFO"


SEVERITY_ORDER = {
    Severity.BLOCKER: 0,
    Severity.MAJOR: 1,
    Severity.MINOR: 2,
    Severity.INFO: 3,
}


@dataclass
class CheckResult:
    """One test case, one verdict."""

    check_id: str
    name: str
    passed: bool
    severity: Severity
    summary: str
    # What a person should actually do about it. A finding without a remedy
    # is a complaint.
    remedy: str = ""
    # Machine-readable specifics: counts, column names, statistics.
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"


@dataclass
class QAReport:
    """Everything one run of the suite found."""

    dataset_name: str
    run_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    results: list[CheckResult] = field(default_factory=list)
    row_counts: dict = field(default_factory=dict)

    def add(self, result: CheckResult) -> None:
        self.results.append(result)

    # -- views -------------------------------------------------------------
    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed]

    def failures_at(self, severity: Severity) -> list[CheckResult]:
        return [r for r in self.failures if r.severity is severity]

    @property
    def sorted_results(self) -> list[CheckResult]:
        """Failures first, worst severity first — the order a person reads in."""
        return sorted(
            self.results,
            key=lambda r: (r.passed, SEVERITY_ORDER[r.severity], r.check_id),
        )

    @property
    def counts(self) -> dict[str, int]:
        out = {"total": len(self.results), "passed": 0, "failed": 0}
        for sev in Severity:
            out[sev.value.lower()] = len(self.failures_at(sev))
        out["passed"] = sum(1 for r in self.results if r.passed)
        out["failed"] = len(self.failures)
        return out

    @property
    def fit_to_train(self) -> bool:
        """
        The gate. One BLOCKER is enough to stop the run.

        Not a score, not a percentage. A dataset with leakage is not 94% fine.
        """
        return not self.failures_at(Severity.BLOCKER)

    # -- output ------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "run_at_utc": self.run_at_utc,
            "row_counts": self.row_counts,
            "counts": self.counts,
            "fit_to_train": self.fit_to_train,
            "results": [r.to_dict() for r in self.sorted_results],
        }

    def write_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path
