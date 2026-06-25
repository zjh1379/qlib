# production/research/_harness.py
"""Shared boilerplate for production/research/ eval runners: env bootstrap, the
OOF -> champion-score contract, and table formatters — so each runner is just its
experiment, not five bands of copied header."""
from __future__ import annotations

import sys as _sys
import sysconfig as _sysconfig
from pathlib import Path

import numpy as np

OOF_FAC = "production/reports/oof_lgbmfac_2021_2026.pkl"
OOF_2MODEL = "production/reports/oof_2model_2021_2026.pkl"
CONFIG = "production/configs/rolling_ensemble.yaml"

_BOOTSTRAPPED = False


def bootstrap() -> None:
    """Idempotent: installed (compiled) qlib ahead of ./qlib source, utf-8 stdout,
    ensure logs/. Call at the top of a runner module, before any qlib import."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    purelib = _sysconfig.get_paths().get("purelib")
    if purelib and purelib not in _sys.path[:1]:
        _sys.path.insert(0, purelib)
    try:
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    Path("logs").mkdir(exist_ok=True)
    _BOOTSTRAPPED = True


def champion_scores():
    """Champion factor-2model score = rebuild_2model(OOF_FAC, OOF_2MODEL)."""
    import pandas as pd
    from production.score_utils import rebuild_2model
    return rebuild_2model(pd.read_pickle(OOF_FAC), pd.read_pickle(OOF_2MODEL))


def pct(x, nd: int = 1) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:+.{nd}%}"


def num(x, nd: int = 2) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"
