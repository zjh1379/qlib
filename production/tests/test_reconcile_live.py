import pytest
from production.research._reconcile_live import decompose_trade


def test_decompose_flags_chasing_and_topk():
    out = decompose_trade(fill_price=10.5, backtest_open=10.0, bt_fwd_ret=-0.03,
                          rank=2, buyable=True, top_k=5)
    assert out["in_topk"] is True
    assert out["entry_premium_pct"] == pytest.approx(0.05)  # paid 5% above open
    assert out["bt_fwd_ret"] == pytest.approx(-0.03)
    assert out["buyable_at_open"] is True


def test_decompose_not_topk_and_unbuyable():
    out = decompose_trade(fill_price=12.0, backtest_open=12.0, bt_fwd_ret=0.08,
                          rank=40, buyable=False, top_k=5)
    assert out["in_topk"] is False
    assert out["entry_premium_pct"] == pytest.approx(0.0)
    assert out["buyable_at_open"] is False
