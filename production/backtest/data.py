# production/backtest/data.py
"""Load realized 1-day open->open returns from qlib for the backtest engine."""
from __future__ import annotations

import pandas as pd

# fwd_ret(d) = open(d+2)/open(d+1) - 1, the realized return of a position
# decided at d, entered next-day open, held one day. Matches the 1d label.
FWD_RET_EXPR = "Ref($open, -2) / Ref($open, -1) - 1"


def init_qlib_from_config(config_path: str = "production/configs/rolling_ensemble.yaml") -> None:
    from production.rolling_train import load_config, init_qlib
    init_qlib(load_config(config_path))


def load_fwd_returns(instruments, start: str, end: str,
                     config_path: str = "production/configs/rolling_ensemble.yaml") -> pd.Series:
    """Returns Series (datetime,instrument)->fwd_ret_1d for the given instruments/date range.
    `instruments` may be a list of qlib codes (e.g. ['SH600000']) or a market string."""
    from qlib.data.dataset.loader import QlibDataLoader
    init_qlib_from_config(config_path)
    loader = QlibDataLoader(config={"feature": ([FWD_RET_EXPR], ["fwd_ret_1d"])})
    df = loader.load(instruments=instruments, start_time=start, end_time=end)
    s = df.iloc[:, 0] if isinstance(df, pd.DataFrame) else df
    if s.index.names[0] == "instrument":
        s = s.swaplevel().sort_index()
    s.index = s.index.set_names(["datetime", "instrument"])
    return s.rename("fwd_ret_1d").dropna()
