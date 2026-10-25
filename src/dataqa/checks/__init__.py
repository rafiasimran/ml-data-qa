"""
The individual test cases.

Each module owns one family of defect and returns CheckResult objects. Keeping
them separate means a check can be read, tested and argued with on its own.
"""
from . import drift, integrity, labels, leakage, schema  # noqa: F401
