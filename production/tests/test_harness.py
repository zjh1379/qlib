import sys
import pytest
from production.research import _harness


def test_pct_formats_and_nan():
    assert _harness.pct(0.123) == "+12.3%"
    assert _harness.pct(-0.05) == "-5.0%"
    assert _harness.pct(float("nan")) == "n/a"
    assert _harness.pct(None) == "n/a"


def test_num_formats_and_nan():
    assert _harness.num(1.2345) == "1.23"
    assert _harness.num(1.2345, 4) == "1.2345"
    assert _harness.num(float("inf")) == "n/a"


def test_bootstrap_idempotent_inserts_purelib():
    import sysconfig
    _harness._BOOTSTRAPPED = False
    _harness.bootstrap()
    purelib = sysconfig.get_paths().get("purelib")
    assert purelib in sys.path
    assert _harness._BOOTSTRAPPED is True
    _harness.bootstrap()  # idempotent, no raise
