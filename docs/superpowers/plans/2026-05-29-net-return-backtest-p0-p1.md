# 净收益回测引擎 + 降换手组合（P0+P1）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个成本感知、持仓周期感知、多 regime 的真回测引擎，并用降换手组合（Daily/FixedPeriod/Banded + 中性化）在扣费净收益口径下超越诚实基线。

**Architecture:** 纯 pandas 的持仓模拟器，把"模型分数（仅排序）"与"真实日频开盘→开盘 1 日收益（记 P&L）"解耦，按再平衡策略逐日记账、仅在换仓日按绝对成交金额计费（含最低 5 元）。引擎消费**已有的** pred.pkl，无需重训模型，因此本计划完全可在现有数据上跑通并测出净收益。

**Tech Stack:** Python 3.x, pandas, numpy, scipy, pytest, qlib（仅 `QlibDataLoader` 取价 + `production.rolling_train.init_qlib`）。Python 解释器：`F:/Tools/Anaconda/envs/qlib/python.exe`。

---

## 关键设计约定（所有任务共享）

- **决策日空间**：在决策日 `d`（收盘后用 ≤d 信息算分数）形成目标权重，次日开盘 `open(d+1)` 成交，持有 1 天的实现收益为 `fwd_ret(d) = open(d+2)/open(d+1) - 1`（即现有 1d 标签 `Ref($open,-2)/Ref($open,-1)-1`）。
- **无前视**：决策只用 `scores.xs(d)`；`fwd_ret(d)` 是 d 之后实现的结果，仅用于记 P&L。
- **成本以人民币计**：每个标的每笔订单 `commission=max(notional*bps/1e4, min_yuan)`，卖出加印花税，双边加过户费与滑点；`notional=|Δw|*nav`，`nav` 初值=`capital`。
- **等权 top-K**；换手 = `0.5*Σ|Δw|`（单边）。
- 索引统一 `(datetime, instrument)` MultiIndex；qlib 返回若为 `(instrument, datetime)` 则 `swaplevel().sort_index()`。

## 文件结构

| 文件 | 职责 |
|---|---|
| `production/backtest/__init__.py` | 包标记 |
| `production/backtest/costs.py` | `CostModel`（佣金/最低费/印花税/过户费/滑点）+ profile 工厂 |
| `production/backtest/rebalance.py` | `RebalancePolicy` 接口 + `Daily`/`FixedPeriod`/`Banded` |
| `production/backtest/metrics_net.py` | 净收益指标 + 多 regime 切分 |
| `production/backtest/engine.py` | `run_backtest` 持仓模拟器 |
| `production/backtest/data.py` | `load_fwd_returns` 取真实 1 日开→开收益（qlib）|
| `production/backtest/run.py` | CLI：单 pred 文件评估 → JSON 报告 |
| `production/backtest/sweep.py` | top_k×H×policy×capital 扫描 → 前沿报告 |
| `production/neutralize.py` | 行业/市值中性化分数 |
| `production/fetch_industry.py` | 抓取 baostock 行业分类（含可测解析函数）|
| `production/tests/test_*.py` | 各模块单测 |

---

## Phase P0 — 可信回测引擎 + 诚实基线

### Task 1: CostModel（成本模型）

**Files:**
- Create: `production/backtest/__init__.py`
- Create: `production/backtest/costs.py`
- Test: `production/tests/test_costs.py`

- [ ] **Step 1: Write the failing test**

```python
# production/tests/test_costs.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_costs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'production.backtest'`

- [ ] **Step 3: Write minimal implementation**

```python
# production/backtest/__init__.py
```

```python
# production/backtest/costs.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_costs.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add production/backtest/__init__.py production/backtest/costs.py production/tests/test_costs.py
git commit -m "feat(backtest): cost model with per-order min + sell-side stamp"
```

---

### Task 2: RebalancePolicy 接口 + Daily

**Files:**
- Create: `production/backtest/rebalance.py`
- Test: `production/tests/test_rebalance.py`

- [ ] **Step 1: Write the failing test**

```python
# production/tests/test_rebalance.py
import pandas as pd
from production.backtest.rebalance import Daily


def test_daily_always_rebalances():
    p = Daily(top_k=2)
    assert p.should_rebalance(0) is True
    assert p.should_rebalance(7) is True


def test_daily_equal_weight_top_k():
    scores = pd.Series({"A": 0.9, "B": 0.5, "C": 0.1})
    w = Daily(top_k=2).target_weights(scores, pd.Series(dtype=float))
    assert set(w.index) == {"A", "B"}
    assert w["A"] == 0.5 and w["B"] == 0.5
    assert w.sum() == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_rebalance.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError: cannot import name 'Daily'`

- [ ] **Step 3: Write minimal implementation**

```python
# production/backtest/rebalance.py
"""Rebalance policies. Each returns equal-weighted long-only target weights."""
from __future__ import annotations

import pandas as pd


def _equal_top_k(scores: pd.Series, k: int) -> pd.Series:
    s = scores.dropna()
    if s.empty or k <= 0:
        return pd.Series(dtype=float)
    top = s.nlargest(k).index
    return pd.Series(1.0 / len(top), index=top)


class RebalancePolicy:
    def should_rebalance(self, step: int) -> bool:
        raise NotImplementedError

    def target_weights(self, scores: pd.Series, current: pd.Series) -> pd.Series:
        raise NotImplementedError


class Daily(RebalancePolicy):
    def __init__(self, top_k: int = 30):
        self.top_k = top_k

    def should_rebalance(self, step: int) -> bool:
        return True

    def target_weights(self, scores: pd.Series, current: pd.Series) -> pd.Series:
        return _equal_top_k(scores, self.top_k)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_rebalance.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add production/backtest/rebalance.py production/tests/test_rebalance.py
git commit -m "feat(backtest): RebalancePolicy interface + Daily top-k"
```

---

### Task 3: 净收益指标 metrics_net

**Files:**
- Create: `production/backtest/metrics_net.py`
- Test: `production/tests/test_metrics_net.py`

- [ ] **Step 1: Write the failing test**

```python
# production/tests/test_metrics_net.py
import numpy as np
import pandas as pd
from production.backtest.metrics_net import net_metrics, net_regime


def _mk_daily(n=252, mean=0.001, std=0.01, turnover=0.1, cost=0.0001, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n)
    net = rng.normal(mean, std, n)
    gross = net + cost
    return pd.DataFrame(
        {"gross": gross, "cost": cost, "net": net,
         "turnover": turnover, "nav": (1 + pd.Series(net, index=dates)).cumprod() * 1e5},
        index=pd.Index(dates, name="datetime"),
    )


def test_net_metrics_keys_and_types():
    out = net_metrics(_mk_daily())
    for k in ["net_cagr", "gross_cagr", "net_ir", "max_drawdown",
              "avg_turnover", "cost_drag_annual", "win_rate", "n_days"]:
        assert k in out
    assert out["n_days"] == 252
    assert out["avg_turnover"] == pytest.approx(0.1, rel=1e-9)


def test_positive_drift_has_positive_cagr():
    out = net_metrics(_mk_daily(mean=0.002, std=0.005, seed=1))
    assert out["net_cagr"] > 0
    assert out["net_ir"] > 0


def test_empty_daily_returns_nan():
    out = net_metrics(pd.DataFrame(columns=["gross", "cost", "net", "turnover", "nav"]))
    assert out["n_days"] == 0
    assert np.isnan(out["net_ir"])


def test_net_regime_splits():
    daily = _mk_daily(n=200)
    segs = net_regime(daily, [("2024-01-01", "2024-03-31"), ("2024-04-01", "2024-12-31")])
    assert len(segs) == 2
    for _, m in segs.items():
        assert "net_ir" in m
```

(Add `import pytest` at top of the test file.)

- [ ] **Step 2: Run test to verify it fails**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_metrics_net.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'production.backtest.metrics_net'`

- [ ] **Step 3: Write minimal implementation**

```python
# production/backtest/metrics_net.py
"""Net-of-cost portfolio metrics computed from the engine's daily ledger."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Multi-regime segments (reused from production/metrics.py conventions).
DEFAULT_REGIMES: list[tuple[str, str]] = [
    ("2018-01-01", "2018-12-31"),   # bear
    ("2019-01-01", "2020-01-31"),   # recovery
    ("2020-02-01", "2021-02-28"),   # covid liquidity
    ("2021-03-01", "2022-12-31"),   # high vol
    ("2023-01-01", "2026-12-31"),   # AI rally / recent
]

_NAN_KEYS = ["net_cagr", "gross_cagr", "net_ir", "max_drawdown",
             "avg_turnover", "cost_drag_annual", "win_rate", "n_days"]


def net_metrics(daily: pd.DataFrame, periods_per_year: int = 252) -> dict:
    if daily is None or daily.empty or "net" not in daily:
        out = {k: float("nan") for k in _NAN_KEYS}
        out["n_days"] = 0
        return out
    r = daily["net"].dropna()
    if r.empty:
        out = {k: float("nan") for k in _NAN_KEYS}
        out["n_days"] = 0
        return out
    n = len(r)
    equity = (1 + r).cumprod()
    gross_equity = (1 + daily["gross"].reindex(r.index).fillna(0.0)).cumprod()

    def _cagr(eq: pd.Series) -> float:
        return float(eq.iloc[-1] ** (periods_per_year / n) - 1) if eq.iloc[-1] > 0 else float("nan")

    ir = float(r.mean() / r.std() * np.sqrt(periods_per_year)) if r.std() > 0 else float("nan")
    dd = float((equity / equity.cummax() - 1).min())
    return {
        "net_cagr": _cagr(equity),
        "gross_cagr": _cagr(gross_equity),
        "net_ir": ir,
        "max_drawdown": dd,
        "avg_turnover": float(daily["turnover"].mean()),
        "cost_drag_annual": float(daily["cost"].mean() * periods_per_year),
        "win_rate": float((r > 0).mean()),
        "n_days": int(n),
    }


def net_regime(daily: pd.DataFrame, segments: list[tuple[str, str]] | None = None,
               periods_per_year: int = 252) -> dict[str, dict]:
    segments = segments or DEFAULT_REGIMES
    out: dict[str, dict] = {}
    idx = pd.to_datetime(daily.index)
    for start, end in segments:
        mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
        sub = daily.loc[mask]
        if sub.empty:
            continue
        out[f"{start}__{end}"] = net_metrics(sub, periods_per_year)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_metrics_net.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add production/backtest/metrics_net.py production/tests/test_metrics_net.py
git commit -m "feat(backtest): net-of-cost metrics + regime split"
```

---

### Task 4: 持仓模拟器 engine

**Files:**
- Create: `production/backtest/engine.py`
- Test: `production/tests/test_engine.py`

- [ ] **Step 1: Write the failing test**

```python
# production/tests/test_engine.py
import pandas as pd
import pytest
from production.backtest.engine import run_backtest
from production.backtest.rebalance import Daily
from production.backtest.costs import CostModel


def _series(d):
    idx = pd.MultiIndex.from_tuples(list(d.keys()), names=["datetime", "instrument"])
    return pd.Series(list(d.values()), index=idx)


def test_cost_reconciliation_first_day():
    # 2 dates, 2 stocks; Daily top_k=1 picks A both days.
    dates = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    scores = _series({(dates[0], "A"): 1.0, (dates[0], "B"): 0.0,
                      (dates[1], "A"): 1.0, (dates[1], "B"): 0.0})
    fwd = _series({(dates[0], "A"): 0.01, (dates[0], "B"): 0.0,
                   (dates[1], "A"): 0.02, (dates[1], "B"): 0.0})
    cm = CostModel(commission_bps=2.5, commission_min_yuan=5.0,
                   stamp_bps=5.0, transfer_bps=0.1, slippage_bps=5.0)
    res = run_backtest(scores, fwd, Daily(top_k=1), cm, capital=100_000.0)
    day0 = res["daily"].iloc[0]
    # Day0: buy A notional=100000 -> cost = max(25,5)+0+1+50 = 76 -> 0.00076
    assert day0["cost"] == pytest.approx(76 / 100_000, rel=1e-9)
    assert day0["turnover"] == pytest.approx(0.5, rel=1e-9)  # 0.5*|+1.0| (B has 0 delta)
    assert day0["gross"] == pytest.approx(0.01, rel=1e-9)
    assert day0["net"] == pytest.approx(0.01 - 0.00076, rel=1e-9)


def test_no_trade_second_day_when_holding_same():
    dates = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    scores = _series({(dates[0], "A"): 1.0, (dates[1], "A"): 1.0})
    fwd = _series({(dates[0], "A"): 0.01, (dates[1], "A"): 0.02})
    res = run_backtest(scores, fwd, Daily(top_k=1), CostModel(), capital=100_000.0)
    # Day1 holds same A -> delta 0 -> cost 0, turnover 0
    assert res["daily"].iloc[1]["cost"] == 0.0
    assert res["daily"].iloc[1]["turnover"] == 0.0
    assert res["daily"].iloc[1]["gross"] == pytest.approx(0.02, rel=1e-9)


def test_iterates_only_dates_with_fwd_ret():
    dates = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")]
    scores = _series({(d, "A"): 1.0 for d in dates})
    # fwd_ret missing the last date (no future price)
    fwd = _series({(dates[0], "A"): 0.01, (dates[1], "A"): 0.02})
    res = run_backtest(scores, fwd, Daily(top_k=1), CostModel(), capital=100_000.0)
    assert len(res["daily"]) == 2  # last date dropped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'production.backtest.engine'`

- [ ] **Step 3: Write minimal implementation**

```python
# production/backtest/engine.py
"""Holding-period-aware portfolio simulator.

Decouples the model SCORE (ranking only) from realized 1-day open->open
returns (P&L). Costs charged in yuan on actual traded notional per order.
"""
from __future__ import annotations

import pandas as pd

from .costs import CostModel
from .rebalance import RebalancePolicy


def run_backtest(
    scores,
    fwd_ret: pd.Series,
    policy: RebalancePolicy,
    cost: CostModel,
    capital: float = 100_000.0,
    score_col: str = "score",
) -> dict:
    """
    scores: Series (datetime,instrument)->score, OR DataFrame with `score_col`.
    fwd_ret: Series (datetime,instrument)->open(d+2)/open(d+1)-1.
    Returns dict: {"daily": DataFrame[gross,cost,net,turnover,nav], "final_nav", "capital"}.
    """
    s = scores[score_col] if isinstance(scores, pd.DataFrame) else scores
    fwd_dates = set(fwd_ret.index.get_level_values("datetime").unique())
    dates = [d for d in sorted(s.index.get_level_values("datetime").unique()) if d in fwd_dates]

    current = pd.Series(dtype=float)
    nav = float(capital)
    rows = []
    for i, d in enumerate(dates):
        s_d = s.xs(d, level="datetime")
        cost_frac = 0.0
        turnover = 0.0
        if policy.should_rebalance(i):
            target = policy.target_weights(s_d, current)
            allidx = current.index.union(target.index)
            cur = current.reindex(allidx).fillna(0.0)
            tgt = target.reindex(allidx).fillna(0.0)
            delta = tgt - cur
            turnover = float(0.5 * delta.abs().sum())
            total_cost = 0.0
            for inst, dw in delta.items():
                if dw == 0:
                    continue
                total_cost += cost.trade_cost(abs(dw) * nav, is_buy=dw > 0)
            cost_frac = total_cost / nav if nav > 0 else 0.0
            current = target
        r_d = fwd_ret.xs(d, level="datetime")
        gross = float((current * r_d.reindex(current.index).fillna(0.0)).sum())
        net = gross - cost_frac
        nav *= (1 + net)
        rows.append({"datetime": d, "gross": gross, "cost": cost_frac,
                     "net": net, "turnover": turnover, "nav": nav})

    daily = pd.DataFrame(rows).set_index("datetime") if rows else \
        pd.DataFrame(columns=["gross", "cost", "net", "turnover", "nav"])
    return {"daily": daily, "final_nav": nav, "capital": float(capital)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_engine.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add production/backtest/engine.py production/tests/test_engine.py
git commit -m "feat(backtest): holding-aware portfolio simulator with yuan costs"
```

---

### Task 5: 真实日收益加载 data.load_fwd_returns

**Files:**
- Create: `production/backtest/data.py`
- Test: `production/tests/test_backtest_data.py`

- [ ] **Step 1: Write the failing test** (structural — no qlib needed)

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_backtest_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'production.backtest.data'`

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_backtest_data.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add production/backtest/data.py production/tests/test_backtest_data.py
git commit -m "feat(backtest): qlib forward-return loader (open-to-open 1d)"
```

---

### Task 6: 回测 CLI run.py（诚实基线报告）

**Files:**
- Create: `production/backtest/run.py`
- Test: `production/tests/test_backtest_run.py`

- [ ] **Step 1: Write the failing test** (tests the pure report-builder, no qlib/CLI)

```python
# production/tests/test_backtest_run.py
import json
import pandas as pd
import pytest
from production.backtest.run import build_report, extract_score_series


def _series(d):
    idx = pd.MultiIndex.from_tuples(list(d.keys()), names=["datetime", "instrument"])
    return pd.Series(list(d.values()), index=idx)


def test_extract_score_series_from_dataframe():
    df = pd.DataFrame({"score": [1.0, 2.0], "other": [9, 9]},
                      index=pd.MultiIndex.from_tuples(
                          [(pd.Timestamp("2024-01-02"), "A"), (pd.Timestamp("2024-01-02"), "B")],
                          names=["datetime", "instrument"]))
    s = extract_score_series(df, "score")
    assert list(s.values) == [1.0, 2.0]


def test_build_report_has_metrics_and_params():
    dates = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    scores = _series({(dates[0], "A"): 1.0, (dates[1], "A"): 1.0})
    fwd = _series({(dates[0], "A"): 0.01, (dates[1], "A"): 0.02})
    rep = build_report(scores, fwd, policy_name="daily", top_k=1, period=5,
                       exit_k=2, capital=100_000.0, profile="small")
    assert "metrics" in rep and "params" in rep
    assert rep["params"]["policy"] == "daily"
    assert rep["metrics"]["n_days"] == 2
    # JSON-serializable
    json.dumps(rep)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_backtest_run.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'production.backtest.run'`

- [ ] **Step 3: Write minimal implementation**

```python
# production/backtest/run.py
"""CLI: evaluate a prediction file under a rebalance policy + realistic costs,
write a net-of-cost JSON scorecard. Usage:

  F:/Tools/Anaconda/envs/qlib/python.exe -m production.backtest.run \
    --pred-file examples/mlruns/.../pred_2026-05-22.pkl \
    --policy banded --top-k 15 --exit-k 30 --capital 100000 \
    --out production/reports/backtest_banded.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .costs import cost_model
from .engine import run_backtest
from .metrics_net import net_metrics, net_regime
from .rebalance import Daily, FixedPeriod, Banded


def extract_score_series(pred, score_col: str = "score") -> pd.Series:
    if isinstance(pred, pd.DataFrame):
        col = score_col if score_col in pred.columns else pred.columns[0]
        return pred[col]
    return pred


def _make_policy(name: str, top_k: int, period: int, exit_k: int):
    if name == "daily":
        return Daily(top_k=top_k)
    if name == "fixed":
        return FixedPeriod(top_k=top_k, period=period)
    if name == "banded":
        return Banded(top_k=top_k, exit_k=exit_k)
    raise ValueError(f"unknown policy {name!r}")


def build_report(scores: pd.Series, fwd_ret: pd.Series, *, policy_name: str,
                 top_k: int, period: int, exit_k: int, capital: float,
                 profile: str) -> dict:
    policy = _make_policy(policy_name, top_k, period, exit_k)
    cm = cost_model(profile)
    res = run_backtest(scores, fwd_ret, policy, cm, capital=capital)
    return {
        "params": {"policy": policy_name, "top_k": top_k, "period": period,
                   "exit_k": exit_k, "capital": capital, "profile": profile},
        "metrics": net_metrics(res["daily"]),
        "regimes": net_regime(res["daily"]),
        "final_nav": res["final_nav"],
        "generated_at": datetime.utcnow().isoformat(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Net-of-cost backtest of a prediction file.")
    ap.add_argument("--pred-file", required=True)
    ap.add_argument("--score-col", default="score")
    ap.add_argument("--policy", default="daily", choices=["daily", "fixed", "banded"])
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--period", type=int, default=5)
    ap.add_argument("--exit-k", type=int, default=60)
    ap.add_argument("--capital", type=float, default=100_000.0)
    ap.add_argument("--profile", default="small", choices=["small", "pro"])
    ap.add_argument("--config", default="production/configs/rolling_ensemble.yaml")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pred = pd.read_pickle(args.pred_file)
    scores = extract_score_series(pred, args.score_col).dropna()
    instruments = sorted(scores.index.get_level_values("instrument").unique())
    dates = scores.index.get_level_values("datetime")
    start, end = str(dates.min().date()), str(dates.max().date())

    from .data import load_fwd_returns
    fwd = load_fwd_returns(instruments, start, end, config_path=args.config)

    rep = build_report(scores, fwd, policy_name=args.policy, top_k=args.top_k,
                       period=args.period, exit_k=args.exit_k, capital=args.capital,
                       profile=args.profile)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}  net_ir={rep['metrics']['net_ir']:.3f}  "
          f"turnover={rep['metrics']['avg_turnover']:.3f}  "
          f"net_cagr={rep['metrics']['net_cagr']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

> NOTE: this task imports `FixedPeriod` and `Banded`, which are added in Tasks 7–8. Run the unit test (Step 4) now — it only exercises `Daily` via `policy_name="daily"`. The `--policy fixed/banded` CLI paths become live after Tasks 7–8.

- [ ] **Step 4: Run test to verify it passes**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_backtest_run.py -v`
Expected: PASS (2 passed). If `ImportError: cannot import name 'FixedPeriod'`, proceed to Task 7/8 first then re-run — but prefer adding empty stubs is NOT allowed; instead reorder: do Tasks 7–8 before Task 6 Step 4.

> To avoid the import ordering issue, **implement Tasks 7 and 8 before running Task 6 Step 4.** Task 6 Steps 1–3 (write test + code) can be done first; its green bar depends on 7–8.

- [ ] **Step 5: Commit**

```bash
git add production/backtest/run.py production/tests/test_backtest_run.py
git commit -m "feat(backtest): CLI report builder (net scorecard + regimes)"
```

---

## Phase P1 — 降换手组合构建

### Task 7: FixedPeriod 策略

**Files:**
- Modify: `production/backtest/rebalance.py`
- Test: `production/tests/test_rebalance.py`

- [ ] **Step 1: Write the failing test (append)**

```python
# append to production/tests/test_rebalance.py
from production.backtest.rebalance import FixedPeriod


def test_fixed_period_rebalances_every_n_steps():
    p = FixedPeriod(top_k=2, period=5)
    assert p.should_rebalance(0) is True
    assert p.should_rebalance(1) is False
    assert p.should_rebalance(5) is True
    assert p.should_rebalance(10) is True


def test_fixed_period_turnover_lower_than_daily_in_engine():
    import pandas as pd
    from production.backtest.engine import run_backtest
    from production.backtest.rebalance import Daily
    from production.backtest.costs import CostModel
    import numpy as np
    dates = pd.bdate_range("2024-01-02", periods=20)
    stocks = [f"S{i}" for i in range(10)]
    idx = pd.MultiIndex.from_product([dates, stocks], names=["datetime", "instrument"])
    g = np.random.default_rng(0)
    scores = pd.Series(g.normal(0, 1, len(idx)), index=idx)
    fwd = pd.Series(g.normal(0, 0.01, len(idx)), index=idx)
    daily = run_backtest(scores, fwd, Daily(top_k=3), CostModel(), 1e5)["daily"]
    fixed = run_backtest(scores, fwd, FixedPeriod(top_k=3, period=5), CostModel(), 1e5)["daily"]
    assert fixed["turnover"].mean() < daily["turnover"].mean()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_rebalance.py -v`
Expected: FAIL with `ImportError: cannot import name 'FixedPeriod'`

- [ ] **Step 3: Write minimal implementation (append to rebalance.py)**

```python
# append to production/backtest/rebalance.py
class FixedPeriod(RebalancePolicy):
    def __init__(self, top_k: int = 30, period: int = 5):
        self.top_k = top_k
        self.period = max(1, period)

    def should_rebalance(self, step: int) -> bool:
        return step % self.period == 0

    def target_weights(self, scores: pd.Series, current: pd.Series) -> pd.Series:
        return _equal_top_k(scores, self.top_k)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_rebalance.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add production/backtest/rebalance.py production/tests/test_rebalance.py
git commit -m "feat(backtest): FixedPeriod policy (rebalance every N steps)"
```

---

### Task 8: Banded（滞后）策略 — 换手杀手

**Files:**
- Modify: `production/backtest/rebalance.py`
- Test: `production/tests/test_rebalance.py`

- [ ] **Step 1: Write the failing test (append)**

```python
# append to production/tests/test_rebalance.py
from production.backtest.rebalance import Banded


def test_banded_keeps_held_name_inside_exit_band():
    p = Banded(top_k=2, exit_k=4)
    # currently hold A,B (equal weight). New scores rank A=1,B=3,C=2,D=4,E=5.
    current = pd.Series({"A": 0.5, "B": 0.5})
    scores = pd.Series({"A": 5.0, "C": 4.0, "B": 3.0, "D": 2.0, "E": 1.0})
    w = p.target_weights(scores, current)
    # A (rank1) and B (rank3 <= exit_k 4) are both kept -> no churn into C.
    assert set(w.index) == {"A", "B"}


def test_banded_drops_held_name_outside_exit_band():
    p = Banded(top_k=2, exit_k=3)
    current = pd.Series({"A": 0.5, "B": 0.5})
    # B falls to rank 4 (> exit_k 3) -> dropped, replaced by best new (C rank2).
    scores = pd.Series({"A": 5.0, "C": 4.0, "D": 3.0, "B": 2.0, "E": 1.0})
    w = p.target_weights(scores, current)
    assert "B" not in w.index
    assert set(w.index) == {"A", "C"}


def test_banded_turnover_lower_than_daily():
    import numpy as np
    from production.backtest.engine import run_backtest
    from production.backtest.rebalance import Daily
    from production.backtest.costs import CostModel
    dates = pd.bdate_range("2024-01-02", periods=30)
    stocks = [f"S{i}" for i in range(20)]
    idx = pd.MultiIndex.from_product([dates, stocks], names=["datetime", "instrument"])
    g = np.random.default_rng(1)
    # persistent-ish scores so banding can hold
    base = {s: g.normal(0, 1) for s in stocks}
    vals = [base[s] + g.normal(0, 0.2) for (_, s) in idx]
    scores = pd.Series(vals, index=idx)
    fwd = pd.Series(g.normal(0, 0.01, len(idx)), index=idx)
    daily = run_backtest(scores, fwd, Daily(top_k=5), CostModel(), 1e5)["daily"]
    banded = run_backtest(scores, fwd, Banded(top_k=5, exit_k=10), CostModel(), 1e5)["daily"]
    assert banded["turnover"].mean() < daily["turnover"].mean()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_rebalance.py -v`
Expected: FAIL with `ImportError: cannot import name 'Banded'`

- [ ] **Step 3: Write minimal implementation (append to rebalance.py)**

```python
# append to production/backtest/rebalance.py
class Banded(RebalancePolicy):
    """Hysteresis: buy a name when rank <= top_k; only sell a held name when
    its rank drops beyond exit_k. Naturally extends holding + cuts turnover."""

    def __init__(self, top_k: int = 30, exit_k: int | None = None):
        self.top_k = top_k
        self.exit_k = exit_k if exit_k is not None else 2 * top_k

    def should_rebalance(self, step: int) -> bool:
        return True

    def target_weights(self, scores: pd.Series, current: pd.Series) -> pd.Series:
        s = scores.dropna()
        if s.empty:
            return pd.Series(dtype=float)
        ranks = s.rank(ascending=False, method="first")
        held = list(current[current > 0].index) if current is not None and not current.empty else []
        keep = [i for i in held if i in ranks.index and ranks[i] <= self.exit_k]
        need = self.top_k - len(keep)
        if need > 0:
            cand = ranks[ranks <= self.top_k].sort_values().index
            for i in cand:
                if i not in keep:
                    keep.append(i)
                    need -= 1
                    if need == 0:
                        break
        keep = keep[: self.top_k]
        if not keep:
            return pd.Series(dtype=float)
        return pd.Series(1.0 / len(keep), index=pd.Index(keep))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_rebalance.py production/tests/test_backtest_run.py -v`
Expected: PASS (rebalance + run report tests all green; Task 6 import now resolves)

- [ ] **Step 5: Commit**

```bash
git add production/backtest/rebalance.py production/tests/test_rebalance.py
git commit -m "feat(backtest): Banded hysteresis policy (turnover killer)"
```

---

### Task 9: 中性化 neutralize

**Files:**
- Create: `production/neutralize.py`
- Test: `production/tests/test_neutralize.py`

- [ ] **Step 1: Write the failing test**

```python
# production/tests/test_neutralize.py
import numpy as np
import pandas as pd
import pytest
from production.neutralize import neutralize


def _scores(dates, stocks, vals):
    idx = pd.MultiIndex.from_product([dates, stocks], names=["datetime", "instrument"])
    return pd.Series(vals, index=idx)


def test_sector_neutral_zeros_each_sector_mean_per_day():
    dates = [pd.Timestamp("2024-01-02")]
    stocks = ["A", "B", "C", "D"]
    s = _scores(dates, stocks, [1.0, 3.0, 10.0, 20.0])
    sector = pd.Series({"A": "x", "B": "x", "C": "y", "D": "y"})
    out = neutralize(s, sector=sector)
    day = out.xs(dates[0], level="datetime")
    # within each sector, mean removed -> sector means are ~0
    assert day[["A", "B"]].mean() == pytest.approx(0.0, abs=1e-9)
    assert day[["C", "D"]].mean() == pytest.approx(0.0, abs=1e-9)


def test_returns_same_index():
    dates = pd.bdate_range("2024-01-02", periods=3)
    stocks = ["A", "B", "C"]
    s = _scores(dates, stocks, np.arange(9, dtype=float))
    out = neutralize(s, size=pd.Series({"A": 1.0, "B": 2.0, "C": 3.0}))
    assert out.index.equals(s.index)
    assert out.name == s.name
```

(Add `import numpy as np` at top.)

- [ ] **Step 2: Run test to verify it fails**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_neutralize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'production.neutralize'`

- [ ] **Step 3: Write minimal implementation**

```python
# production/neutralize.py
"""Cross-sectional score neutralization (sector demean + size regression)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def neutralize(scores: pd.Series, sector: pd.Series | None = None,
               size: pd.Series | None = None) -> pd.Series:
    """Per-datetime: subtract sector mean, then regress out size (rank).
    `sector`/`size` are instrument-indexed Series. Returns same-index Series."""
    name = scores.name or "score"
    parts = []
    for d, g in scores.groupby(level="datetime"):
        x = g.copy()
        inst = x.index.get_level_values("instrument")
        if sector is not None:
            sec = pd.Series(sector.reindex(inst).values, index=x.index)
            grp_mean = x.groupby(sec.values).transform("mean")
            x = x - pd.Series(grp_mean.values, index=x.index)
        if size is not None:
            sz = pd.Series(size.reindex(inst).values, index=x.index).rank()
            zc = sz - sz.mean()
            xc = x - x.mean()
            denom = float((zc * zc).sum())
            beta = float((xc * zc).sum() / denom) if denom > 0 else 0.0
            x = x - beta * zc
        parts.append(x)
    return pd.concat(parts).rename(name)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_neutralize.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add production/neutralize.py production/tests/test_neutralize.py
git commit -m "feat: cross-sectional sector/size neutralization"
```

---

### Task 10: 行业分类抓取 fetch_industry

**Files:**
- Create: `production/fetch_industry.py`
- Test: `production/tests/test_fetch_industry.py`

- [ ] **Step 1: Write the failing test** (tests the pure parser, not the network)

```python
# production/tests/test_fetch_industry.py
import pandas as pd
from production.fetch_industry import parse_industry_rows, to_qlib_code


def test_to_qlib_code():
    assert to_qlib_code("sh.600000") == "SH600000"
    assert to_qlib_code("sz.000001") == "SZ000001"


def test_parse_industry_rows():
    # baostock query_stock_industry row format: [updateDate, code, code_name, industry, industryClassification]
    rows = [
        ["2024-01-01", "sh.600000", "浦发银行", "银行", "申万一级"],
        ["2024-01-01", "sz.300750", "宁德时代", "电池", "申万一级"],
        ["2024-01-01", "sz.000002", "万科A", "", "申万一级"],  # empty industry
    ]
    df = parse_industry_rows(rows)
    assert list(df.columns) == ["instrument", "industry"]
    m = dict(zip(df["instrument"], df["industry"]))
    assert m["SH600000"] == "银行"
    assert m["SZ300750"] == "电池"
    # empty industry filled with "UNKNOWN"
    assert m["SZ000002"] == "UNKNOWN"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_fetch_industry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'production.fetch_industry'`

- [ ] **Step 3: Write minimal implementation**

```python
# production/fetch_industry.py
"""Fetch SW industry classification from baostock; save instrument->industry map.

Usage:
  F:/Tools/Anaconda/envs/qlib/python.exe -m production.fetch_industry \
    --out production/cache/industry_map.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def to_qlib_code(bs_code: str) -> str:
    """'sh.600000' -> 'SH600000'."""
    market, num = bs_code.split(".")
    return f"{market.upper()}{num}"


def parse_industry_rows(rows: list[list[str]]) -> pd.DataFrame:
    """Parse baostock query_stock_industry rows into instrument->industry.
    Row layout: [updateDate, code, code_name, industry, industryClassification]."""
    recs = []
    for r in rows:
        if len(r) < 4:
            continue
        code = to_qlib_code(r[1])
        industry = (r[3] or "").strip() or "UNKNOWN"
        recs.append({"instrument": code, "industry": industry})
    return pd.DataFrame(recs, columns=["instrument", "industry"])


def fetch_industry_map() -> pd.DataFrame:
    import baostock as bs
    lg = bs.login()
    try:
        rs = bs.query_stock_industry()
        if rs.error_code != "0":
            raise RuntimeError(f"query_stock_industry failed: {rs.error_msg}")
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
    finally:
        bs.logout()
    return parse_industry_rows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch baostock SW industry map.")
    ap.add_argument("--out", default="production/cache/industry_map.parquet")
    args = ap.parse_args()
    df = fetch_industry_map()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {out} rows={len(df)} industries={df['industry'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_fetch_industry.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the fetcher once (network) + Commit**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m production.fetch_industry --out production/cache/industry_map.parquet`
Expected: prints `wrote ... rows=<~5000> industries=<~30>`

```bash
git add production/fetch_industry.py production/tests/test_fetch_industry.py
git commit -m "feat: baostock SW industry fetch + parser (for neutralization)"
```

---

### Task 11: 参数扫描 sweep（top_k × H × policy × capital）

**Files:**
- Create: `production/backtest/sweep.py`
- Test: `production/tests/test_sweep.py`

- [ ] **Step 1: Write the failing test**

```python
# production/tests/test_sweep.py
import numpy as np
import pandas as pd
from production.backtest.sweep import run_sweep


def _mk():
    dates = pd.bdate_range("2024-01-02", periods=40)
    stocks = [f"S{i}" for i in range(25)]
    idx = pd.MultiIndex.from_product([dates, stocks], names=["datetime", "instrument"])
    g = np.random.default_rng(0)
    base = {s: g.normal(0, 1) for s in stocks}
    scores = pd.Series([base[s] + g.normal(0, 0.3) for (_, s) in idx], index=idx)
    fwd = pd.Series(g.normal(0.0005, 0.01, len(idx)), index=idx)
    return scores, fwd


def test_run_sweep_grid_shape_and_columns():
    scores, fwd = _mk()
    grid = run_sweep(
        scores, fwd,
        policies=["daily", "banded"],
        top_ks=[5, 10],
        periods=[5],
        capitals=[50_000, 100_000],
        profile="small",
    )
    # 2 policies x 2 top_k x 2 capital (period only matters for 'fixed') = 8 rows
    assert len(grid) == 8
    for col in ["policy", "top_k", "capital", "net_ir", "avg_turnover", "net_cagr"]:
        assert col in grid.columns


def test_run_sweep_banded_turnover_below_daily_on_average():
    scores, fwd = _mk()
    grid = run_sweep(scores, fwd, policies=["daily", "banded"], top_ks=[10],
                     periods=[5], capitals=[100_000], profile="small")
    t_daily = grid[grid.policy == "daily"]["avg_turnover"].mean()
    t_banded = grid[grid.policy == "banded"]["avg_turnover"].mean()
    assert t_banded < t_daily
```

- [ ] **Step 2: Run test to verify it fails**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_sweep.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'production.backtest.sweep'`

- [ ] **Step 3: Write minimal implementation**

```python
# production/backtest/sweep.py
"""Grid sweep over policy x top_k x holding-period x capital -> net-return frontier.

CLI:
  F:/Tools/Anaconda/envs/qlib/python.exe -m production.backtest.sweep \
    --pred-file <pred.pkl> --out production/reports/sweep.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .costs import cost_model
from .engine import run_backtest
from .metrics_net import net_metrics
from .rebalance import Daily, FixedPeriod, Banded


def _policy(name: str, top_k: int, period: int):
    if name == "daily":
        return Daily(top_k=top_k)
    if name == "fixed":
        return FixedPeriod(top_k=top_k, period=period)
    if name == "banded":
        return Banded(top_k=top_k, exit_k=2 * top_k)
    raise ValueError(name)


def run_sweep(scores: pd.Series, fwd_ret: pd.Series, *, policies, top_ks,
              periods, capitals, profile: str = "small") -> pd.DataFrame:
    cm = cost_model(profile)
    rows = []
    for pol in policies:
        # period only varies the grid for the 'fixed' policy
        per_list = periods if pol == "fixed" else [periods[0]]
        for k in top_ks:
            for per in per_list:
                for cap in capitals:
                    res = run_backtest(scores, fwd_ret, _policy(pol, k, per), cm, capital=cap)
                    m = net_metrics(res["daily"])
                    rows.append({"policy": pol, "top_k": k, "period": per,
                                 "capital": cap, **m})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Backtest parameter sweep.")
    ap.add_argument("--pred-file", required=True)
    ap.add_argument("--score-col", default="score")
    ap.add_argument("--profile", default="small", choices=["small", "pro"])
    ap.add_argument("--config", default="production/configs/rolling_ensemble.yaml")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from .run import extract_score_series
    from .data import load_fwd_returns

    pred = pd.read_pickle(args.pred_file)
    scores = extract_score_series(pred, args.score_col).dropna()
    instruments = sorted(scores.index.get_level_values("instrument").unique())
    dates = scores.index.get_level_values("datetime")
    fwd = load_fwd_returns(instruments, str(dates.min().date()), str(dates.max().date()),
                           config_path=args.config)

    grid = run_sweep(scores, fwd,
                     policies=["daily", "fixed", "banded"],
                     top_ks=[5, 10, 15, 20, 30],
                     periods=[1, 2, 3, 5],
                     capitals=[50_000, 100_000, 300_000, 1_000_000],
                     profile=args.profile)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.sort_values("net_ir", ascending=False).to_csv(out, index=False, encoding="utf-8-sig")
    best = grid.loc[grid["net_ir"].idxmax()]
    print(f"wrote {out} rows={len(grid)}")
    print(f"BEST net_ir: policy={best['policy']} top_k={best['top_k']} "
          f"capital={best['capital']} net_ir={best['net_ir']:.3f} "
          f"turnover={best['avg_turnover']:.3f} net_cagr={best['net_cagr']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_sweep.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add production/backtest/sweep.py production/tests/test_sweep.py
git commit -m "feat(backtest): policy/top_k/holding/capital sweep -> net-return frontier"
```

---

### Task 12: 端到端验证 + 诚实基线对比 + 文档

**Files:**
- Create: `production/reports/` JSON/CSV outputs (generated, not committed unless small)
- Create: `docs/superpowers/specs/2026-05-29-net-return-results.md` (results summary)
- Modify: `docs/superpowers/specs/2026-05-29-shortterm-net-return-design.md` (link results)

- [ ] **Step 1: Locate a real pred file**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -c "import glob; print('\n'.join(sorted(glob.glob('examples/mlruns/**/pred_*.pkl', recursive=True))[-5:]))"`
Expected: prints recent pooled ensemble pred files (e.g. `pred_2026-05-22.pkl`). Pick the most recent.

- [ ] **Step 2: Run the honest baseline (Daily) under realistic costs**

Run (substitute the path from Step 1):
```
F:/Tools/Anaconda/envs/qlib/python.exe -m production.backtest.run \
  --pred-file <PRED> --policy daily --top-k 30 --capital 100000 --profile small \
  --out production/reports/honest_baseline_daily.json
```
Expected: prints `net_ir=... turnover=~0.4x net_cagr=...` — the honest (likely much lower than the bogus 54%) baseline.

- [ ] **Step 3: Run the sweep to find the net-return optimum**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m production.backtest.sweep --pred-file <PRED> --out production/reports/sweep.csv`
Expected: prints BEST row (expected: a Banded or small-top_k config with much lower turnover and higher net_ir than Daily).

- [ ] **Step 4: Run the chosen improved config + verify acceptance**

Run (use BEST policy/top_k from Step 3, e.g. banded/15):
```
F:/Tools/Anaconda/envs/qlib/python.exe -m production.backtest.run \
  --pred-file <PRED> --policy banded --top-k 15 --exit-k 30 --capital 100000 \
  --profile small --out production/reports/improved_banded.json
```
Verify against spec §8 acceptance:
- improved `avg_turnover` ≤ 0.5 × baseline `avg_turnover`
- improved `net_ir` > baseline `net_ir`
- all `regimes[*].net_ir` present (positivity checked in summary)

Run the comparison helper:
```
F:/Tools/Anaconda/envs/qlib/python.exe -c "import json; b=json.load(open('production/reports/honest_baseline_daily.json')); i=json.load(open('production/reports/improved_banded.json')); print('turnover', b['metrics']['avg_turnover'], '->', i['metrics']['avg_turnover']); print('net_ir', b['metrics']['net_ir'], '->', i['metrics']['net_ir']); print('net_cagr', b['metrics']['net_cagr'], '->', i['metrics']['net_cagr'])"
```
Expected: turnover down ≥50%, net_ir up.

- [ ] **Step 5: Write results summary + run full test suite + commit**

Write `docs/superpowers/specs/2026-05-29-net-return-results.md` with: honest baseline numbers, sweep best config, improved-vs-baseline deltas (turnover, net_ir, net_cagr), per-capital recommendation ("你这个资金量该持几只、多久换").

Run full backtest suite:
```
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_costs.py production/tests/test_rebalance.py production/tests/test_metrics_net.py production/tests/test_engine.py production/tests/test_backtest_data.py production/tests/test_backtest_run.py production/tests/test_neutralize.py production/tests/test_fetch_industry.py production/tests/test_sweep.py -v
```
Expected: all PASS.

```bash
git add docs/superpowers/specs/2026-05-29-net-return-results.md docs/superpowers/specs/2026-05-29-shortterm-net-return-design.md production/reports/honest_baseline_daily.json production/reports/improved_banded.json production/reports/sweep.csv
git commit -m "docs(backtest): honest baseline vs improved net-return results (P0+P1)"
```

---

## Self-Review

**1. Spec coverage（对照 design §3–§10）：**
- §3 成本感知回测引擎 → Task 1 (costs)、Task 4 (engine)、Task 5 (fwd ret)、Task 3 (metrics)。✓
- §3.2 最低费 + capital 感知 → Task 1 + engine 用 `|Δw|*nav` 计 notional。✓
- §3.3 多 regime → Task 3 `net_regime` + Task 12 验收。✓ （长窗 LGBM 回填属 P2 计划，本计划用现有 pred 在其覆盖窗口评估——design §3.3 的长回填在 P2 plan 执行。）
- §4.1 再平衡策略 Daily/FixedPeriod/Banded → Task 2/7/8。✓
- §4.2 持仓周期扫描 → Task 11 sweep（periods 1/2/3/5）。✓
- §4.3 中性化 + 行业数据依赖 → Task 9 + Task 10。✓
- §4.4 小资金/最低费 top_k 分析 → Task 11 capitals 维度 + Task 12 结论。✓
- §7 测试（无前视/成本数学/换手单调/净值对账）→ Task 1/4/7/8 断言。✓
- §8 验收 → Task 12 Step 4。✓
- §10 算力安全 → 本计划不训练模型（仅消费现有 pred），天然无卡死风险。✓
- §5 短线因子（P2）→ 不在本计划，单列 P2 plan。✓（范围拆分已说明）

**2. Placeholder scan:** 无 TBD/TODO；每个代码步骤含完整代码；CLI 与测试均给出确切命令与期望。Task 6 的导入顺序坑已显式标注（先做 7/8 再绿 6）。✓

**3. Type consistency:**
- `CostModel.trade_cost(notional, is_buy)` 在 engine Task 4 与测试一致。✓
- `cost_model(profile, **overrides)` 在 run/sweep 一致。✓
- 策略接口 `should_rebalance(step)` / `target_weights(scores, current)` 三实现一致，engine 调用一致。✓
- `run_backtest(scores, fwd_ret, policy, cost, capital, score_col)` → `{"daily","final_nav","capital"}`；run/sweep/metrics 消费 `daily[gross,cost,net,turnover,nav]` 一致。✓
- `net_metrics(daily)` / `net_regime(daily, segments)` 键名与 Task 3 定义、Task 12 校验一致。✓
- `load_fwd_returns(instruments, start, end, config_path)` 在 run/sweep 调用一致；`FWD_RET_EXPR` 常量被测试钉死。✓
- `extract_score_series(pred, score_col)` 在 run/sweep 共用。✓
- `neutralize(scores, sector, size)` 与测试一致。✓
- `parse_industry_rows` / `to_qlib_code` 与测试一致。✓

无遗漏，无类型漂移。

---

## 后续（独立 plan）
- **P2 plan**：扩展 baostock 抓取 `amount,turn` + dump；短线因子（OVNGAP*/TURN*/limit）；`AlphaShortTerm` handler；长窗 LGBM OOF 回填（watchdog 下，coarse walk-forward）；因子增量评估（本计划的引擎直接复用）。
- **P3 / P4**：见 design §11。
