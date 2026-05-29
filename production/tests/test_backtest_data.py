# production/tests/test_backtest_data.py
import inspect
from production.backtest import data as bt_data


def test_load_fwd_returns_signature_and_expr():
    # Function exists with expected params.
    sig = inspect.signature(bt_data.load_fwd_returns)
    assert list(sig.parameters)[:3] == ["instruments", "start", "end"]
    # The forward-return expression must be open(d+2)/open(d+1)-1 (no lookahead in decision,
    # realized after d). Guard the exact string to prevent silent off-by-one regressions.
    assert bt_data.FWD_RET_EXPR == "Ref($open, -2) / Ref($open, -1) - 1"
