# production/backtest/data.py
"""Load realized 1-day open->open returns from qlib for the backtest engine."""
from __future__ import annotations

# --- sys.path fixup: same pattern as production/rolling_train.py ---
# Running `-m production.backtest.run` from the repo root puts the uncompiled
# qlib/ source on sys.path before site-packages; insert site-packages first so
# the installed qlib (with compiled .pyd extensions) wins.
import sys as _sys
import sysconfig as _sysconfig

_PURELIB = _sysconfig.get_paths().get("purelib")
if _PURELIB and _PURELIB not in _sys.path[:1]:
    _sys.path.insert(0, _PURELIB)
# --- end fixup ---

import pandas as pd

# fwd_ret(d) = open(d+2)/open(d+1) - 1, the realized return of a position
# decided at d, entered next-day open, held one day. Matches the 1d label.
FWD_RET_EXPR = "Ref($open, -2) / Ref($open, -1) - 1"


def init_qlib_from_config(config_path: str = "production/configs/rolling_ensemble.yaml") -> None:
    from pathlib import Path
    from production.rolling_train import load_config, init_qlib
    init_qlib(load_config(Path(config_path)))


def load_fwd_returns(instruments, start: str, end: str,
                     config_path: str = "production/configs/rolling_ensemble.yaml") -> pd.Series:
    """Returns Series (datetime,instrument)->fwd_ret_1d for the given instruments/date range.
    `instruments` may be a list of qlib codes (e.g. ['SH600000']) or a market string."""
    from production.qlib_features import load_series
    return load_series(instruments, start, end, FWD_RET_EXPR, "fwd_ret_1d",
                       config_path=config_path, dropna=True)
