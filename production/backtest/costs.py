"""A-share retail trading cost model (commission w/ per-order minimum,
stamp tax on sells only, transfer fee + slippage both sides)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostModel:
    commission_bps: float = 2.5      # 万2.5
    commission_min_yuan: float = 5.0  # 最低5元/笔
    stamp_bps: float = 5.0           # 印花税，仅卖出 (0.05%)
    transfer_bps: float = 0.1        # 过户费，双边
    slippage_bps: float = 5.0        # 滑点，双边

    def trade_cost(self, notional: float, is_buy: bool) -> float:
        """Cost in yuan for a single instrument order of |notional| yuan."""
        notional = abs(float(notional))
        if notional <= 0:
            return 0.0
        commission = max(notional * self.commission_bps / 1e4, self.commission_min_yuan)
        stamp = 0.0 if is_buy else notional * self.stamp_bps / 1e4
        transfer = notional * self.transfer_bps / 1e4
        slippage = notional * self.slippage_bps / 1e4
        return commission + stamp + transfer + slippage


PROFILES: dict[str, dict] = {
    "small": dict(commission_bps=2.5, commission_min_yuan=5.0,
                  stamp_bps=5.0, transfer_bps=0.1, slippage_bps=5.0),
    "pro": dict(commission_bps=1.0, commission_min_yuan=0.0,
                stamp_bps=5.0, transfer_bps=0.1, slippage_bps=5.0),
}


def cost_model(profile: str = "small", **overrides) -> CostModel:
    base = dict(PROFILES[profile])
    base.update(overrides)
    return CostModel(**base)
