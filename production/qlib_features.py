# production/qlib_features.py
"""One deep seam for loading adjusted features from qlib as a clean
(datetime, instrument) frame. Collapses the swaplevel + set_names + column
normalization ritual that 6+ call sites each re-implemented — inconsistently
(some df.iloc[:,0], some df.columns=names; the latter divergence shipped as a bug).
Domain-named loaders (load_fwd_returns, load_entry_ohlc, daily_open_adj, ...) stay as
thin wrappers over this; the qlib-quirk handling lives here once."""
from __future__ import annotations

# Force installed (compiled) qlib ahead of the uncompiled ./qlib source tree.
import sys as _sys
import sysconfig as _sysconfig
_PURELIB = _sysconfig.get_paths().get("purelib")
if _PURELIB and _PURELIB not in _sys.path[:1]:
    _sys.path.insert(0, _PURELIB)

import pandas as pd

DEFAULT_CONFIG = "production/configs/rolling_ensemble.yaml"


def _normalize(df: pd.DataFrame, names: list) -> pd.DataFrame:
    """Pure: raw QlibDataLoader frame -> clean (datetime,instrument) DataFrame whose
    columns are exactly `names`. QlibDataLoader labels columns by the feature
    EXPRESSION (or a MultiIndex level) and indexes by (instrument, datetime); we force
    `names` by position and put datetime first. No qlib import."""
    if df is None:
        raise ValueError("qlib loader returned None")
    df = df.copy()
    if df.shape[1] != len(names):
        raise ValueError(f"expected {len(names)} columns, got {df.shape[1]}: {list(df.columns)}")
    df.columns = list(names)
    if df.index.names and df.index.names[0] == "instrument":
        df = df.swaplevel()
    df.index = df.index.set_names(["datetime", "instrument"])
    return df.sort_index()


def load_features(instruments, start: str, end: str, fields: list, names: list,
                  *, config_path: str = DEFAULT_CONFIG, dropna_subset=None) -> pd.DataFrame:
    """Load `fields` (qlib expressions) for `instruments` over [start,end] as a clean
    (datetime,instrument) DataFrame with columns `names`; optional dropna on a subset."""
    from qlib.data.dataset.loader import QlibDataLoader
    from production.backtest.data import init_qlib_from_config
    init_qlib_from_config(config_path)
    raw = QlibDataLoader(config={"feature": (list(fields), list(names))}).load(
        instruments=instruments, start_time=start, end_time=end)
    out = _normalize(raw, names)
    if dropna_subset is not None:
        out = out.dropna(subset=list(dropna_subset))
    return out


def load_series(instruments, start: str, end: str, field: str, name: str,
                *, config_path: str = DEFAULT_CONFIG, dropna: bool = False) -> pd.Series:
    """Single-field convenience -> Series named `name`, (datetime,instrument) index."""
    s = load_features(instruments, start, end, [field], [name], config_path=config_path)[name]
    return s.dropna() if dropna else s
