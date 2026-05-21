import sys
from pathlib import Path

# When pytest is invoked from the qlib repo root, the local source tree
# (which contains a partially-built `qlib/` package without the compiled
# `_libs.rolling` C extension) shadows the installed qlib. Temporarily
# strip the repo root from sys.path so the FIRST `import qlib` resolves to
# the conda-env install, then restore sys.path so `production.*` modules
# remain importable.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_saved_sys_path = list(sys.path)
sys.path[:] = [p for p in sys.path if Path(p).resolve() != _REPO_ROOT]
import qlib  # noqa: F401  # noqa: E402 — populate sys.modules with the installed qlib
sys.path[:] = _saved_sys_path

import pytest  # noqa: E402

from production.custom_handler import (  # noqa: E402
    Alpha158_OpenH,
    Alpha360_OpenH,
)


def test_alpha158_openh_5d_label_formula():
    handler = Alpha158_OpenH.__new__(Alpha158_OpenH)
    handler.horizon_days = 5
    fields, names = handler.get_label_config()
    assert names == ["LABEL0"]
    assert fields == ["Ref($open, -6) / Ref($open, -1) - 1"]


def test_alpha158_openh_1d_label_formula():
    handler = Alpha158_OpenH.__new__(Alpha158_OpenH)
    handler.horizon_days = 1
    fields, names = handler.get_label_config()
    assert fields == ["Ref($open, -2) / Ref($open, -1) - 1"]


def test_alpha158_openh_20d_label_formula():
    handler = Alpha158_OpenH.__new__(Alpha158_OpenH)
    handler.horizon_days = 20
    fields, names = handler.get_label_config()
    assert fields == ["Ref($open, -21) / Ref($open, -1) - 1"]


def test_alpha360_openh_same_label_formula():
    handler = Alpha360_OpenH.__new__(Alpha360_OpenH)
    handler.horizon_days = 5
    fields, names = handler.get_label_config()
    assert fields == ["Ref($open, -6) / Ref($open, -1) - 1"]
