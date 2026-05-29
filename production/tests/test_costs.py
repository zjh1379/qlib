import pytest
from production.backtest.costs import CostModel, cost_model


def test_commission_min_dominates_small_notional():
    cm = CostModel(commission_bps=2.5, commission_min_yuan=5.0,
                   stamp_bps=5.0, transfer_bps=0.1, slippage_bps=5.0)
    # buy 1000 yuan: commission = max(1000*2.5/1e4=0.25, 5)=5; transfer=0.01; slip=0.5
    assert cm.trade_cost(1000.0, is_buy=True) == pytest.approx(5 + 0.01 + 0.5, rel=1e-9)


def test_commission_bps_dominates_large_notional_and_stamp_on_sell():
    cm = CostModel(commission_bps=2.5, commission_min_yuan=5.0,
                   stamp_bps=5.0, transfer_bps=0.1, slippage_bps=5.0)
    # sell 100000: commission=max(25,5)=25; stamp=50; transfer=1; slip=50
    assert cm.trade_cost(100000.0, is_buy=False) == pytest.approx(25 + 50 + 1 + 50, rel=1e-9)
    # buy 100000: no stamp
    assert cm.trade_cost(100000.0, is_buy=True) == pytest.approx(25 + 0 + 1 + 50, rel=1e-9)


def test_zero_notional_is_free():
    cm = CostModel()
    assert cm.trade_cost(0.0, is_buy=True) == 0.0


def test_pro_profile_has_no_min_and_lower_bps():
    cm = cost_model("pro")
    assert cm.commission_min_yuan == 0.0
    assert cm.commission_bps == 1.0
    # buy 1000: commission=1000*1/1e4=0.1 (no min); transfer=0.01; slip=0.5
    assert cm.trade_cost(1000.0, is_buy=True) == pytest.approx(0.1 + 0.01 + 0.5, rel=1e-9)
