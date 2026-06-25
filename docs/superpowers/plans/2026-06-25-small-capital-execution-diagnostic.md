# 小资金操作点诊断 (A1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在用户真实的 ¥10k、单只 2–3 手 约束下,诚实测出当前冠军模型还剩多少"买得进、扣完费"的净收益,并裁决"单票 top-1/2 vs ETF 择时"哪条路有正期望。

**Architecture:** 四块诊断,纯逻辑放进可单测的稳定模块(`production/backtest/`),数据装配放进 research runner(`production/research/`)。复用现有 `build_report`/`run_backtest`/`score_utils.rebuild_2model`/`metrics_net`,并复用 `intraday.entry_rules.limit_up_price` 做涨停判定(DRY)。

**Tech Stack:** Python 3.10 (`F:/Tools/Anaconda/envs/qlib/python.exe`), pandas/numpy, pytest, qlib (cn_data_bs 日线)。

---

## 执行环境(关键 · 多 worktree 约束)

- 代码在 **worktree** 编辑+提交:`E:/Projects/qlib/.claude/worktrees/cool-pasteur-962838`,分支 `claude/cool-pasteur-962838`。
- **单元测试是纯函数**(合成数据,无需 qlib 数据/OOF),直接在 worktree 跑:
  `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/<file>.py -v`
- **数据回测 runner** 需要 OOF pkl(只在主库、被 gitignore)。config 的 `provider_uri: ~/.qlib/...` 是 home 路径(与 cwd 无关),所以只需把两个 OOF 拷进 worktree 即可在 worktree 直接跑(见 Task 9 Step 0):
  - `production/reports/oof_lgbmfac_2021_2026.pkl`、`production/reports/oof_2model_2021_2026.pkl`
- 命令一律用 `-X utf8`,从 worktree 根目录运行 `-m production.research.<name>`。

## File Structure

**新增(稳定、可单测)**
- `production/backtest/executability.py` — 涨停可成交 mask + 门控选股 + 选择性偏差分解(Block 2 核心)
- `production/backtest/etf_timing.py` — 单资产 ETF 择时微型 sim(Block 4 核心)

**修改(稳定、可单测)**
- `production/backtest/costs.py` — 加 `etf` 成本 profile(免印花/免过户)
- `production/backtest/metrics_net.py` — 加 `tail_stats`(左尾/方差诊断)
- `production/research/_eval_topk_sweep.py` — top_k 加 {1,2,3}、加 ¥10k 档、加左尾列(Block 1 runner)

**新增(research runner,不单测,Task 9 真跑)**
- `production/research/_eval_executability.py` — Block 2 装配:门控 vs 未门控 + 偏差分解
- `production/research/_eval_etf_timing.py` — Block 4 装配
- `production/research/_reconcile_live.py` — Block 3:两笔真实交易拆 信号 vs 执行(含可单测的 `decompose_trade`)

**新增(本地数据,gitignored)**
- `production/reports/live_trades.csv` — 用户填的真实成交

**结果(Task 9 跑完写)**
- `docs/superpowers/specs/2026-06-25-small-capital-execution-diagnostic-results.md`

---

## Task 1: `etf` 成本 profile (Block 4 支撑)

**Files:**
- Modify: `production/backtest/costs.py:28-33` (PROFILES dict)
- Test: `production/tests/test_costs.py`

- [ ] **Step 1: Write the failing test**

在 `production/tests/test_costs.py` 末尾追加:

```python
def test_etf_profile_no_stamp_no_transfer():
    cm = cost_model("etf")
    assert cm.stamp_bps == 0.0
    assert cm.transfer_bps == 0.0
    assert cm.commission_min_yuan == 5.0
    # sell 10000 ETF: commission=max(10000*2.5/1e4=2.5, 5)=5; stamp=0; transfer=0; slip=5 => 10
    assert cm.trade_cost(10000.0, is_buy=False) == pytest.approx(5 + 0 + 0 + 5, rel=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_costs.py::test_etf_profile_no_stamp_no_transfer -v`
Expected: FAIL with `KeyError: 'etf'`

- [ ] **Step 3: Add the etf profile**

在 `production/backtest/costs.py` 的 `PROFILES` dict 内(`"pro"` 之后)加一项:

```python
    "etf": dict(commission_bps=2.5, commission_min_yuan=5.0,
                stamp_bps=0.0, transfer_bps=0.0, slippage_bps=5.0),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_costs.py -v`
Expected: PASS (4 旧 + 1 新)

- [ ] **Step 5: Commit**

```bash
git add production/backtest/costs.py production/tests/test_costs.py
git commit -m "feat(backtest): add etf cost profile (no stamp/transfer)"
```

---

## Task 2: `tail_stats` 左尾/方差诊断 (Block 1 支撑)

**Files:**
- Modify: `production/backtest/metrics_net.py` (append function)
- Test: `production/tests/test_metrics_net.py`

- [ ] **Step 1: Write the failing test**

在 `production/tests/test_metrics_net.py` 末尾追加(文件顶部已 `import pandas as pd`、`import pytest`;若无则补):

```python
def test_tail_stats_basic():
    from production.backtest.metrics_net import tail_stats
    s = pd.Series([-0.10, -0.02, 0.01, 0.03, 0.20])
    out = tail_stats(s)
    assert out["worst"] == pytest.approx(-0.10)
    assert out["neg_period_pct"] == pytest.approx(0.4)
    assert out["n"] == 5
    assert out["ret_p10"] == pytest.approx(float(s.quantile(0.10)))


def test_tail_stats_empty_is_nan():
    from production.backtest.metrics_net import tail_stats
    out = tail_stats(pd.Series([], dtype=float))
    assert out["n"] == 0
    assert out["worst"] != out["worst"]  # nan
```

- [ ] **Step 2: Run test to verify it fails**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_metrics_net.py::test_tail_stats_basic -v`
Expected: FAIL with `ImportError: cannot import name 'tail_stats'`

- [ ] **Step 3: Implement `tail_stats`**

在 `production/backtest/metrics_net.py` 末尾追加:

```python
def tail_stats(net) -> dict:
    """Left-tail / variance diagnostics for a per-period net-return Series — how
    brutal a concentrated (top-1/2) book is. `net` = engine daily ledger 'net' col
    (or any return Series)."""
    r = pd.Series(net).dropna()
    if r.empty:
        return {"ret_p05": float("nan"), "ret_p10": float("nan"),
                "ret_std": float("nan"), "neg_period_pct": float("nan"),
                "worst": float("nan"), "n": 0}
    return {
        "ret_p05": float(r.quantile(0.05)),
        "ret_p10": float(r.quantile(0.10)),
        "ret_std": float(r.std(ddof=1)) if len(r) > 1 else float("nan"),
        "neg_period_pct": float((r < 0).mean()),
        "worst": float(r.min()),
        "n": int(len(r)),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_metrics_net.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add production/backtest/metrics_net.py production/tests/test_metrics_net.py
git commit -m "feat(backtest): add tail_stats (left-tail/variance diagnostics)"
```

---

## Task 3: `executability.py` — 涨停可成交 + 门控 + 偏差分解 (Block 2 核心)

**Files:**
- Create: `production/backtest/executability.py`
- Test: `production/tests/test_executability.py`

- [ ] **Step 1: Write the failing tests**

Create `production/tests/test_executability.py`:

```python
import pandas as pd, pytest
from production.backtest.executability import (
    buyable_mask, gate_scores, selection_bias_split)


def _ohlc(rows):
    # rows: (date_str, instrument, entry_open, prev_close)
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(d), i) for d, i, *_ in rows], names=["datetime", "instrument"])
    return pd.DataFrame({"entry_open": [r[2] for r in rows],
                         "prev_close": [r[3] for r in rows]}, index=idx)


def test_buyable_mask_main_board_10pct():
    # SH600000 main 10%: cap=11.0. open 10.5<11 -> buyable; open 11.0==cap -> not
    o = _ohlc([("2024-01-02", "SH600000", 10.5, 10.0),
               ("2024-01-02", "SZ000001", 11.0, 10.0)])
    m = buyable_mask(o)
    assert bool(m.xs("2024-01-02", level="datetime")["SH600000"]) is True
    assert bool(m.xs("2024-01-02", level="datetime")["SZ000001"]) is False


def test_buyable_mask_chinext_20pct():
    # SZ300750 ChiNext 20%: cap=12.0; open 11.5 buyable (would fail on a main-board cap)
    o = _ohlc([("2024-01-02", "SZ300750", 11.5, 10.0)])
    assert bool(buyable_mask(o).iloc[0]) is True


def test_gate_scores_drops_unbuyable_promoting_next_rank():
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-02"), i) for i in ["A", "B", "C"]],
        names=["datetime", "instrument"])
    scores = pd.Series([0.9, 0.8, 0.7], index=idx)
    buyable = pd.Series([False, True, True], index=idx)  # A unbuyable
    g = gate_scores(scores, buyable)
    assert "A" not in g.index.get_level_values("instrument")
    assert g.xs("2024-01-02", level="datetime").idxmax() == "B"  # next rank promoted


def test_gate_scores_keeps_unknown():
    idx = pd.MultiIndex.from_tuples([(pd.Timestamp("2024-01-02"), "A")],
                                    names=["datetime", "instrument"])
    scores = pd.Series([0.9], index=idx)
    g = gate_scores(scores, pd.Series(dtype=bool))  # no info -> keep
    assert len(g) == 1


def test_selection_bias_split_detects_missed_winners():
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-02"), i) for i in ["A", "B", "C"]],
        names=["datetime", "instrument"])
    scores = pd.Series([0.9, 0.8, 0.1], index=idx)
    fwd = pd.Series([0.10, 0.01, 0.0], index=idx)   # A +10% (unbuyable), B +1% (buyable)
    buyable = pd.Series([False, True, True], index=idx)
    out = selection_bias_split(scores, fwd, buyable, top_k=2, period=1)
    assert out["n_picks"] == 2 and out["n_unbuyable"] == 1
    assert out["unbuyable_mean_ret"] == pytest.approx(0.10)
    assert out["buyable_mean_ret"] == pytest.approx(0.01)
    assert out["edge_missed"] == pytest.approx(0.09)  # winners we miss
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_executability.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'production.backtest.executability'`

- [ ] **Step 3: Implement `executability.py`**

Create `production/backtest/executability.py`:

```python
# production/backtest/executability.py
"""A股 entry-open buyability. A name whose entry-day OPEN gaps to its 涨停 ceiling
can't be bought at open. Used to (a) gate top-k selection to buyable names and
(b) decompose whether the unbuyable (gapped-up) names are the winners we miss."""
from __future__ import annotations

# Force installed qlib ahead of the uncompiled ./qlib tree (only load_entry_ohlc
# touches qlib; the pure helpers below don't).
import sys as _sys
import sysconfig as _sysconfig
_PURELIB = _sysconfig.get_paths().get("purelib")
if _PURELIB and _PURELIB not in _sys.path[:1]:
    _sys.path.insert(0, _PURELIB)

import numpy as np
import pandas as pd

from production.intraday.entry_rules import limit_up_price


def buyable_mask(ohlc: pd.DataFrame) -> pd.Series:
    """ohlc indexed (datetime, instrument) with columns [entry_open, prev_close]
    (entry_open = open of entry session d+1; prev_close = close of decision day d).
    True when entry open is strictly below the name's 涨停 ceiling (buy-at-open fills)."""
    idx = ohlc.index
    insts = idx.get_level_values("instrument")
    caps = pd.Series(
        [limit_up_price(i, pc) for i, pc in zip(insts, ohlc["prev_close"].to_numpy())],
        index=idx)
    return (ohlc["entry_open"] < caps - 1e-9).rename("buyable")


def gate_scores(scores: pd.Series, buyable: pd.Series) -> pd.Series:
    """Drop (datetime,instrument) entries that aren't buyable-at-open so downstream
    nlargest rolls to the next buyable rank. Unknown (missing) names are kept."""
    s = scores.dropna()
    if buyable is None or len(buyable) == 0:
        return s
    keep = buyable.reindex(s.index).fillna(True).astype(bool)
    return s[keep]


def selection_bias_split(scores: pd.Series, fwd_ret: pd.Series, buyable: pd.Series,
                         top_k: int, period: int) -> dict:
    """On each rebalance day (every `period` steps) take the UNGATED top-k by score,
    split into buyable-at-open vs unbuyable (gapped-up), collect realized open->open
    fwd returns. edge_missed = mean(unbuyable) - mean(buyable): >0 means the winners
    are exactly the ones a live trader can't buy."""
    s = scores.dropna()
    dates = sorted(s.index.get_level_values("datetime").unique())
    fwd_dates = set(fwd_ret.index.get_level_values("datetime").unique())
    buy_ret, miss_ret = [], []
    n_pick = n_miss = 0
    for i, d in enumerate(dates):
        if i % period != 0 or d not in fwd_dates:
            continue
        cross = s.xs(d, level="datetime").sort_values(ascending=False)
        r_d = fwd_ret.xs(d, level="datetime")
        try:
            b_d = buyable.xs(d, level="datetime")
        except KeyError:
            b_d = pd.Series(dtype=bool)
        for inst in list(cross.index[:top_k]):
            n_pick += 1
            ret = r_d.get(inst)
            if ret is None or pd.isna(ret):
                continue
            if bool(b_d.get(inst, True)):
                buy_ret.append(float(ret))
            else:
                miss_ret.append(float(ret)); n_miss += 1

    def _m(a):
        return float(np.mean(a)) if a else float("nan")
    return {
        "n_picks": n_pick, "n_unbuyable": n_miss,
        "unbuyable_pct": (n_miss / n_pick) if n_pick else 0.0,
        "buyable_mean_ret": _m(buy_ret), "unbuyable_mean_ret": _m(miss_ret),
        "buyable_n": len(buy_ret), "unbuyable_n": len(miss_ret),
        "edge_missed": _m(miss_ret) - _m(buy_ret),
    }


def load_entry_ohlc(instruments, start: str, end: str,
                    config_path: str = "production/configs/rolling_ensemble.yaml") -> pd.DataFrame:
    """Per decision day d, load the entry session's open/high/low + decision-day close
    (涨停 base). Aligned to d like load_fwd_returns (entry at d+1):
      entry_open=Ref($open,-1), entry_high=Ref($high,-1), entry_low=Ref($low,-1),
      prev_close=$close. Integration helper (qlib); not unit-tested."""
    from qlib.data.dataset.loader import QlibDataLoader
    from production.backtest.data import init_qlib_from_config
    init_qlib_from_config(config_path)
    fields = ["Ref($open,-1)", "Ref($high,-1)", "Ref($low,-1)", "$close"]
    names = ["entry_open", "entry_high", "entry_low", "prev_close"]
    df = QlibDataLoader(config={"feature": (fields, names)}).load(
        instruments=instruments, start_time=start, end_time=end)
    if df.index.names[0] == "instrument":
        df = df.swaplevel().sort_index()
    df.index = df.index.set_names(["datetime", "instrument"])
    return df.dropna(subset=["entry_open", "prev_close"])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_executability.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add production/backtest/executability.py production/tests/test_executability.py
git commit -m "feat(backtest): executability mask + gated selection + selection-bias split"
```

---

## Task 4: `etf_timing.py` — 单资产 ETF 择时 sim (Block 4 核心)

**Files:**
- Create: `production/backtest/etf_timing.py`
- Test: `production/tests/test_etf_timing.py`

- [ ] **Step 1: Write the failing tests**

Create `production/tests/test_etf_timing.py`:

```python
import pandas as pd, pytest
from production.backtest.etf_timing import simulate_etf_timing
from production.backtest.costs import cost_model


def test_full_exposure_tracks_etf_minus_flip_cost():
    dates = pd.bdate_range("2024-01-02", periods=4)
    etf = pd.Series([0.01, 0.01, 0.01, 0.01], index=dates)
    exp = pd.Series([1.0, 1.0, 1.0, 1.0], index=dates)
    led = simulate_etf_timing(etf, exp, cost_model("etf"), capital=10_000.0)
    assert led["turnover"].iloc[0] == pytest.approx(0.0)  # pos starts 0 (exposure shifted)
    assert led["turnover"].iloc[1] == pytest.approx(1.0)  # flip 0->1 (one-time cost)
    assert led["net"].iloc[2] == pytest.approx(0.01)      # invested, no flip, no cost


def test_zero_exposure_is_flat():
    dates = pd.bdate_range("2024-01-02", periods=3)
    etf = pd.Series([0.05, -0.05, 0.05], index=dates)
    exp = pd.Series([0.0, 0.0, 0.0], index=dates)
    led = simulate_etf_timing(etf, exp, cost_model("etf"))
    assert led["net"].abs().sum() == pytest.approx(0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_etf_timing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'production.backtest.etf_timing'`

- [ ] **Step 3: Implement `etf_timing.py`**

Create `production/backtest/etf_timing.py`:

```python
# production/backtest/etf_timing.py
"""Tiny single-asset ETF market-timing sim: hold one broad ETF, scale exposure in
[0,1] by the trend overlay, charge ETF cost only when exposure flips. NOT the
cross-sectional stock engine — a one-asset daily ledger compatible with net_metrics."""
from __future__ import annotations

import pandas as pd

from production.backtest.costs import CostModel


def simulate_etf_timing(etf_ret: pd.Series, exposure: pd.Series,
                        cost: CostModel, capital: float = 10_000.0) -> pd.DataFrame:
    """etf_ret: daily ETF return by datetime. exposure: target weight [0,1] by datetime
    (decided from trailing data). Position(d) = exposure(d-1) (acted next day, no
    lookahead); pnl(d)=pos(d)*etf_ret(d); cost on |pos change| notional when flipping.
    Returns daily ledger [gross,cost,net,turnover,nav]."""
    etf_ret = etf_ret.sort_index()
    pos = exposure.reindex(etf_ret.index).ffill().shift(1).fillna(0.0).clip(0.0, 1.0)
    rows = []
    nav = float(capital)
    prev = 0.0
    for d in etf_ret.index:
        p = float(pos.loc[d])
        gross = p * float(etf_ret.loc[d])
        dpos = abs(p - prev)
        cost_yuan = cost.trade_cost(dpos * nav, is_buy=(p >= prev)) if dpos > 1e-9 else 0.0
        cost_frac = cost_yuan / nav if nav > 0 else 0.0
        net = gross - cost_frac
        nav *= (1 + net)
        rows.append({"datetime": d, "gross": gross, "cost": cost_frac,
                     "net": net, "turnover": dpos, "nav": nav})
        prev = p
    return pd.DataFrame(rows).set_index("datetime")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_etf_timing.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add production/backtest/etf_timing.py production/tests/test_etf_timing.py
git commit -m "feat(backtest): single-asset etf market-timing sim"
```

---

## Task 5: `_reconcile_live.py` + `decompose_trade` (Block 3)

**Files:**
- Create: `production/research/_reconcile_live.py`
- Create: `production/reports/live_trades.csv` (template, gitignored)
- Test: `production/tests/test_reconcile_live.py`

- [ ] **Step 1: Write the failing tests**

Create `production/tests/test_reconcile_live.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_reconcile_live.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'production.research._reconcile_live'`

- [ ] **Step 3: Implement `_reconcile_live.py`**

Create `production/research/_reconcile_live.py`:

```python
# production/research/_reconcile_live.py
"""Decompose each real-money trade into SIGNAL (did the model rank it top-k? would the
backtest's open->open have profited?) vs EXECUTION (did you pay above the backtest's
entry open = chase? was it a 涨停 gap = backtest open fill unattainable?). Seeds the
forward-test journal.

Run: F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m production.research._reconcile_live \
  --trades production/reports/live_trades.csv > logs/reconcile_live.log 2>&1
"""
import sys as _sys, sysconfig as _sysconfig
_P = _sysconfig.get_paths().get("purelib")
if _P and _P not in _sys.path[:1]:
    _sys.path.insert(0, _P)
try:
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import argparse
import json
from pathlib import Path
import pandas as pd

OOF_FAC = "production/reports/oof_lgbmfac_2021_2026.pkl"
OOF_2MODEL = "production/reports/oof_2model_2021_2026.pkl"
CONFIG = "production/configs/rolling_ensemble.yaml"


def decompose_trade(fill_price, backtest_open, bt_fwd_ret, rank, buyable, top_k=5) -> dict:
    """entry_premium = fill/open - 1 (>0 => chased above the open the backtest assumes).
    in_topk = was it a model top-k pick (signal). bt_fwd_ret = backtest open->open 1d
    return on the decision day (signal-quality proxy). buyable = open below 涨停 cap
    (False => backtest's open fill was unattainable)."""
    has_open = backtest_open is not None and backtest_open == backtest_open and backtest_open != 0
    return {
        "rank": rank,
        "in_topk": (rank is not None and rank <= top_k),
        "bt_fwd_ret": bt_fwd_ret,
        "entry_premium_pct": (fill_price / backtest_open - 1) if has_open else float("nan"),
        "buyable_at_open": (None if buyable is None else bool(buyable)),
    }


def _rank_on(scores: pd.Series, date, inst) -> "int | None":
    try:
        cross = scores.xs(date, level="datetime").dropna().sort_values(ascending=False)
    except KeyError:
        return None
    order = list(cross.index)
    return order.index(inst) + 1 if inst in order else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", default="production/reports/live_trades.csv")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", default="logs/reconcile_live.json")
    args = ap.parse_args()

    tr = pd.read_csv(args.trades, dtype={"instrument": str})
    tr["trade_date"] = pd.to_datetime(tr["trade_date"])

    from production.score_utils import rebuild_2model
    from production.backtest.data import load_fwd_returns
    from production.backtest.executability import load_entry_ohlc, buyable_mask

    scores = rebuild_2model(pd.read_pickle(OOF_FAC), pd.read_pickle(OOF_2MODEL))
    insts = sorted(tr["instrument"].unique())
    start = str(tr["trade_date"].min().date())
    end = str((tr["trade_date"].max() + pd.Timedelta(days=20)).date())
    fwd = load_fwd_returns(insts, start, end, config_path=CONFIG)
    ohlc = load_entry_ohlc(insts, start, end, config_path=CONFIG)
    buyable = buyable_mask(ohlc)

    rows = []
    for _, t in tr.iterrows():
        d, inst = t["trade_date"], t["instrument"]
        bo = ohlc["entry_open"].get((d, inst), float("nan"))
        dec = decompose_trade(
            fill_price=float(t["fill_price"]),
            backtest_open=float(bo),
            bt_fwd_ret=float(fwd.get((d, inst), float("nan"))),
            rank=_rank_on(scores, d, inst),
            buyable=(bool(buyable.get((d, inst))) if (d, inst) in buyable.index else None),
            top_k=args.top_k)
        dec.update({"trade_date": str(d.date()), "instrument": inst,
                    "fill_price": float(t["fill_price"]),
                    "entry_timing": t.get("entry_timing", ""), "note": t.get("note", "")})
        rows.append(dec)
        print(f"{inst} {str(d.date())} [{dec.get('note','')}]: rank={dec['rank']} "
              f"in_top{args.top_k}={dec['in_topk']} buyable={dec['buyable_at_open']} "
              f"bt_fwd_1d={dec['bt_fwd_ret']:+.2%} entry_premium={dec['entry_premium_pct']:+.2%}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_reconcile_live.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Create the CSV template**

Create `production/reports/live_trades.csv` (用户随后填真实 日期/代码/买入价;`entry_timing` ∈ {open,intraday}):

```csv
trade_date,instrument,side,fill_price,shares,entry_timing,note
2026-06-01,SH600000,buy,0.0,200,open,环保(占位-请替换为真实日期/代码/买入价)
2026-06-10,SH600000,buy,0.0,200,intraday,黄金(占位-请替换为真实日期/代码/买入价)
```

- [ ] **Step 6: Commit**

```bash
git add production/research/_reconcile_live.py production/tests/test_reconcile_live.py
git commit -m "feat(research): reconcile live trades into signal-vs-execution"
```

(`production/reports/live_trades.csv` 被 gitignore,不入库——本地填。)

---

## Task 6: 扩展 `_eval_topk_sweep.py` — top_k {1,2,3} + ¥10k 档 + 左尾列 (Block 1)

**Files:**
- Modify: `production/research/_eval_topk_sweep.py:32-36` (constants), `:52-57` (`_daily`), `:76-129` (`main`)

> 该 runner 跑真数据,无单测;Step 末尾做"导入冒烟",真跑在 Task 9。

- [ ] **Step 1: 改常量(行 32-36)**

把:

```python
OOF_FAC = "production/reports/oof_lgbmfac_2021_2026.pkl"
OOF_2MODEL = "production/reports/oof_2model_2021_2026.pkl"
CONFIG = "production/configs/rolling_ensemble.yaml"
PERIOD, CAPITAL, PROFILE = 5, 100_000.0, "small"
TOP_KS = [5, 10, 15, 20, 30]
```

改为:

```python
OOF_FAC = "production/reports/oof_lgbmfac_2021_2026.pkl"
OOF_2MODEL = "production/reports/oof_2model_2021_2026.pkl"
CONFIG = "production/configs/rolling_ensemble.yaml"
PERIOD, PROFILE = 5, "small"
CAPITALS = [10_000.0, 100_000.0]
TOP_KS = [1, 2, 3, 5, 10, 15, 20, 30]
```

- [ ] **Step 2: 改 `_daily` 加 capital 参数(行 52-57)**

把:

```python
def _daily(scores, fwd, k):
    from production.backtest.engine import run_backtest
    from production.backtest.rebalance import FixedPeriod
    from production.backtest.costs import cost_model
    return run_backtest(scores, fwd, FixedPeriod(top_k=k, period=PERIOD),
                        cost_model(PROFILE), capital=CAPITAL)["daily"]
```

改为:

```python
def _daily(scores, fwd, k, capital):
    from production.backtest.engine import run_backtest
    from production.backtest.rebalance import FixedPeriod
    from production.backtest.costs import cost_model
    return run_backtest(scores, fwd, FixedPeriod(top_k=k, period=PERIOD),
                        cost_model(PROFILE), capital=capital)["daily"]
```

- [ ] **Step 3: 替换整个 `main()`(行 76 起到文件末 `raise SystemExit(main())` 之前)**

```python
def main() -> int:
    Path("logs").mkdir(exist_ok=True)
    from production.score_utils import score_of as _score_of, rebuild_2model as _rebuild_2model
    from production.backtest.metrics_net import net_metrics, tail_stats
    fac = pd.read_pickle(OOF_FAC)
    two = pd.read_pickle(OOF_2MODEL)
    factor_2m = _rebuild_2model(fac, two)
    base_2m = _score_of(two)

    insts = sorted(set(factor_2m.index.get_level_values("instrument")) |
                   set(base_2m.index.get_level_values("instrument")))
    dts = pd.DatetimeIndex(sorted(set(factor_2m.index.get_level_values("datetime")) |
                                  set(base_2m.index.get_level_values("datetime"))))
    start, end = str(dts.min().date()), str(dts.max().date())
    from production.backtest.data import load_fwd_returns
    fwd = load_fwd_returns(insts, start, end, config_path=CONFIG)
    years = sorted({d.year for d in dts})

    out = {}
    for capital in CAPITALS:
        hdr = (f"{'top_k':>5} {'net_cagr':>9} {'net_ir':>7} {'max_dd':>8} {'Calmar':>7} "
               f"{'turnov':>7} {'cost/yr':>8} {'win':>6} {'p10':>8} {'std':>7} {'neg%':>6} "
               f"{'neg_yr':>6} {'t':>6} {'p':>7}")
        print(f"\nFACTOR-2MODEL — top_k sweep (fixed/5d, Y{capital:,.0f}, {PROFILE} cost)")
        print("=" * len(hdr)); print(hdr); print("-" * len(hdr))
        cap_out = {}
        for k in TOP_KS:
            fd = _daily(factor_2m, fwd, k, capital)
            bd = _daily(base_2m, fwd, k, capital)
            m = net_metrics(fd)
            cal = _calmar(m)
            ts = tail_stats(fd["net"])
            yr = {}
            for y in years:
                sub = fd.loc[pd.to_datetime(fd.index).year == y]
                yr[y] = net_metrics(sub)["net_cagr"]
            neg = sum(1 for y in years if np.isfinite(yr[y]) and yr[y] < 0)
            t, p = _paired_t(fd["net"], bd["net"])
            cap_out[k] = {"metrics": m, "calmar": cal, "tail": ts, "per_year": yr,
                          "neg_years": neg, "t": t, "p": p}
            print(f"{k:>5} {_pct(m['net_cagr']):>9} {_num(m['net_ir']):>7} "
                  f"{_pct(m['max_drawdown']):>8} {_num(cal):>7} {_num(m['avg_turnover'],3):>7} "
                  f"{_pct(m['cost_drag_annual']):>8} {_pct(m['win_rate']):>6} "
                  f"{_pct(ts['ret_p10']):>8} {_num(ts['ret_std'],4):>7} "
                  f"{_pct(ts['neg_period_pct']):>6} {neg:>6} {_num(t):>6} {_num(p,4):>7}")
        out[f"capital_{int(capital)}"] = cap_out

    Path("logs/eval_topk_sweep_summary.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    print("\nwrote logs/eval_topk_sweep_summary.json")
    print("(p10/std/neg% = per-DAY net left-tail; t/p = paired daily-net vs baseline at that k)")
    return 0
```

> 注:`tail_stats` 作用在引擎**日频** net 序列上(日级左尾),足以显示 top-1 方差;per-5日块 左尾为可选精化。

- [ ] **Step 4: 导入冒烟(不跑数据)**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -c "import production.research._eval_topk_sweep as m; print(m.TOP_KS, m.CAPITALS)"`
Expected: 打印 `[1, 2, 3, 5, 10, 15, 20, 30] [10000.0, 100000.0]`,无异常

- [ ] **Step 5: Commit**

```bash
git add production/research/_eval_topk_sweep.py
git commit -m "feat(research): topk sweep adds {1,2,3} + ¥10k pass + left-tail cols"
```

---

## Task 7: `_eval_executability.py` runner (Block 2 装配)

**Files:**
- Create: `production/research/_eval_executability.py`

- [ ] **Step 1: Implement the runner**

Create `production/research/_eval_executability.py`:

```python
# production/research/_eval_executability.py
"""涨停 可成交性 + 选择性偏差 (Block 2). Ungated vs gated (skip names whose entry
open gaps to the 涨停 ceiling) net metrics at top_k {1,2,3,5}/¥10k, plus the
buyable-vs-unbuyable realized-return split (does the live trader miss the winners?).

Run: F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m production.research._eval_executability \
  > logs/eval_executability.log 2>&1
"""
import sys as _sys, sysconfig as _sysconfig
_P = _sysconfig.get_paths().get("purelib")
if _P and _P not in _sys.path[:1]:
    _sys.path.insert(0, _P)
try:
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import json
from pathlib import Path
import pandas as pd

OOF_FAC = "production/reports/oof_lgbmfac_2021_2026.pkl"
OOF_2MODEL = "production/reports/oof_2model_2021_2026.pkl"
CONFIG = "production/configs/rolling_ensemble.yaml"
PERIOD, CAPITAL, PROFILE = 5, 10_000.0, "small"
TOP_KS = [1, 2, 3, 5]


def main() -> int:
    Path("logs").mkdir(exist_ok=True)
    from production.score_utils import rebuild_2model
    from production.backtest.data import load_fwd_returns
    from production.backtest.executability import (
        load_entry_ohlc, buyable_mask, gate_scores, selection_bias_split)
    from production.backtest.run import build_report

    scores = rebuild_2model(pd.read_pickle(OOF_FAC), pd.read_pickle(OOF_2MODEL))
    insts = sorted(scores.index.get_level_values("instrument").unique())
    dts = scores.index.get_level_values("datetime")
    start, end = str(dts.min().date()), str(dts.max().date())

    fwd = load_fwd_returns(insts, start, end, config_path=CONFIG)
    ohlc = load_entry_ohlc(insts, start, end, config_path=CONFIG)
    buyable = buyable_mask(ohlc)
    gated = gate_scores(scores, buyable)

    out = {"ungated": {}, "gated": {}, "bias": {}}
    print(f"{'top_k':>5} {'ungated':>9} {'gated':>9} {'unbuy%':>7} {'edge_missed':>12}")
    print("-" * 46)
    for k in TOP_KS:
        u = build_report(scores, fwd_ret=fwd, policy_name="fixed", top_k=k, period=PERIOD,
                         exit_k=2 * k, capital=CAPITAL, profile=PROFILE, config_path=CONFIG)
        g = build_report(gated, fwd_ret=fwd, policy_name="fixed", top_k=k, period=PERIOD,
                         exit_k=2 * k, capital=CAPITAL, profile=PROFILE, config_path=CONFIG)
        bias = selection_bias_split(scores, fwd, buyable, top_k=k, period=PERIOD)
        out["ungated"][k] = u["metrics"]; out["gated"][k] = g["metrics"]; out["bias"][k] = bias
        print(f"{k:>5} {u['metrics']['net_cagr']:>+9.2%} {g['metrics']['net_cagr']:>+9.2%} "
              f"{bias['unbuyable_pct']:>7.1%} {bias['edge_missed']:>+12.2%}")
    Path("logs/eval_executability.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    print("wrote logs/eval_executability.json")
    print("(edge_missed>0 => the unbuyable gapped-up picks are the winners you can't buy)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: 导入冒烟**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -c "import production.research._eval_executability as m; print(m.TOP_KS)"`
Expected: 打印 `[1, 2, 3, 5]`,无异常

- [ ] **Step 3: Commit**

```bash
git add production/research/_eval_executability.py
git commit -m "feat(research): executability runner (gated vs ungated + bias split)"
```

---

## Task 8: `_eval_etf_timing.py` runner (Block 4 装配)

**Files:**
- Create: `production/research/_eval_etf_timing.py`

- [ ] **Step 1: Implement the runner**

Create `production/research/_eval_etf_timing.py`:

```python
# production/research/_eval_etf_timing.py
"""ETF market-timing arm: hold a broad ETF proxy, time it with the P3 trend overlay,
compare net to the single-stock sweep. v1 ETF proxy = synthetic equal-weight universe
(market.py); refinement = real index/ETF NAV.

Run: F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m production.research._eval_etf_timing \
  > logs/eval_etf_timing.log 2>&1
"""
import sys as _sys, sysconfig as _sysconfig
_P = _sysconfig.get_paths().get("purelib")
if _P and _P not in _sys.path[:1]:
    _sys.path.insert(0, _P)
try:
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import json
from pathlib import Path
import pandas as pd

OOF_2MODEL = "production/reports/oof_2model_2021_2026.pkl"
CONFIG = "production/configs/rolling_ensemble.yaml"
CAPITAL = 10_000.0
MA_WINDOW, BAND = 60, 0.10


def main() -> int:
    Path("logs").mkdir(exist_ok=True)
    from production.score_utils import score_of
    from production.backtest.market import load_market_proxy
    from production.backtest.regime import compute_exposure
    from production.backtest.etf_timing import simulate_etf_timing
    from production.backtest.costs import cost_model
    from production.backtest.metrics_net import net_metrics, tail_stats

    two = pd.read_pickle(OOF_2MODEL)
    s = score_of(two)
    insts = sorted(s.index.get_level_values("instrument").unique())
    dts = s.index.get_level_values("datetime")
    start, end = str(dts.min().date()), str(dts.max().date())

    mkt_close = load_market_proxy(insts, start, end, config_path=CONFIG)
    etf_ret = mkt_close.pct_change().dropna()
    exposure = compute_exposure(mkt_close, method="trend_ma", ma_window=MA_WINDOW, band=BAND)

    out = {}
    bh = simulate_etf_timing(etf_ret, pd.Series(1.0, index=etf_ret.index),
                             cost_model("etf"), capital=CAPITAL)
    tm = simulate_etf_timing(etf_ret, exposure, cost_model("etf"), capital=CAPITAL)
    for name, led in [("etf_buyhold", bh), ("etf_timed", tm)]:
        m = net_metrics(led)
        ts = tail_stats(led["net"])
        out[name] = {"metrics": m, "tail": ts}
        print(f"{name:>12}: net_cagr={m['net_cagr']:+.2%} maxDD={m['max_drawdown']:+.2%} "
              f"IR={m['net_ir']:.2f} neg_day%={ts['neg_period_pct']:.1%}")
    Path("logs/eval_etf_timing.json").write_text(json.dumps(out, indent=2, default=float),
                                                 encoding="utf-8")
    print("wrote logs/eval_etf_timing.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: 导入冒烟**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -c "import production.research._eval_etf_timing as m; print(m.MA_WINDOW, m.BAND)"`
Expected: 打印 `60 0.1`,无异常

- [ ] **Step 3: Commit**

```bash
git add production/research/_eval_etf_timing.py
git commit -m "feat(research): etf-timing arm (synthetic broad-market + trend overlay)"
```

---

## Task 9: 真跑 + 汇总对照表 + 结果文档

**Files:**
- Create: `docs/superpowers/specs/2026-06-25-small-capital-execution-diagnostic-results.md`

> 数据回测;在 worktree 跑(需先拷 OOF)。所有命令从 worktree 根目录执行。

- [ ] **Step 0: 全量单测 + 拷 OOF 进 worktree**

```bash
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_costs.py production/tests/test_metrics_net.py production/tests/test_executability.py production/tests/test_etf_timing.py production/tests/test_reconcile_live.py -v
cp E:/Projects/qlib/production/reports/oof_lgbmfac_2021_2026.pkl production/reports/
cp E:/Projects/qlib/production/reports/oof_2model_2021_2026.pkl production/reports/
```
Expected: 全测 PASS;两个 pkl 出现在 worktree `production/reports/`。

- [ ] **Step 1: Block 1 — top_k 下扫**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m production.research._eval_topk_sweep > logs/eval_topk_sweep.log 2>&1`
Expected: `logs/eval_topk_sweep_summary.json` 写出;**回归锚**:`capital_100000` 的 `top_k=5` net_cagr ≈ **+31%**(±2pp)。记录 top_k=1/2/3 在 ¥10k 下的 net_cagr / Calmar / maxDD / p10 / neg%。

- [ ] **Step 2: Block 2 — 涨停可成交 + 偏差**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m production.research._eval_executability > logs/eval_executability.log 2>&1`
Expected: `logs/eval_executability.json` 写出;记录每个 top_k 的 ungated→gated net_cagr、`unbuyable_pct`、`edge_missed`(>0 = 赢家被错过)。

- [ ] **Step 3: Block 4 — ETF 对照**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m production.research._eval_etf_timing > logs/eval_etf_timing.log 2>&1`
Expected: `logs/eval_etf_timing.json` 写出;记录 etf_buyhold vs etf_timed 的 net_cagr / maxDD / IR。

- [ ] **Step 4: Block 3 — 复盘两笔(需用户先填 CSV)**

先让用户把 `production/reports/live_trades.csv` 的占位行替换成真实 日期/代码(qlib 格式如 `SH600000`)/买入价/`entry_timing`。然后:
Run: `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m production.research._reconcile_live --trades production/reports/live_trades.csv > logs/reconcile_live.log 2>&1`
Expected: `logs/reconcile_live.json`;每笔打印 rank / in_topk / buyable / bt_fwd_1d / entry_premium。
> 若用户暂无法提供细节:跳过本步,在结果文档标注 "Block 3 待真实成交回填",其余裁决不受阻塞。

- [ ] **Step 5: 写结果文档**

Create `docs/superpowers/specs/2026-06-25-small-capital-execution-diagnostic-results.md`,按下列骨架填入实测数(房风格同 `*-results.md`):

```markdown
# 小资金操作点诊断 (A1) · 结果 (2026-06-25)

## 裁决(一句话)
[在 ¥10k 下,单票 top-1/2 门控扣费后 net_cagr = X% / Calmar = Y;ETF 择时 = Z%。结论:走 ___ 路线。]

## Block 1 — 操作点下扫(¥10k vs ¥100k,fixed/5d,small cost)
| capital | top_k | net_cagr | Calmar | maxDD | p10(日) | neg% | cost/yr |
|--|--|--|--|--|--|--|--|
[填 logs/eval_topk_sweep_summary.json;含 ¥100k/top5 ≈+31% 回归锚]

## Block 2 — 涨停可成交 + 选择性偏差(¥10k)
| top_k | ungated net_cagr | gated net_cagr | unbuyable% | edge_missed |
|--|--|--|--|--|
[填 logs/eval_executability.json;edge_missed>0 = 赢家被错过的实锤]

## Block 3 — 两笔真实交易复盘
[填 logs/reconcile_live.json:环保/黄金 各自 rank / in_topk / buyable / entry_premium → 信号 vs 执行归因;或标注 待回填]

## Block 4 — ETF 择时对照(¥10k)
| 策略 | net_cagr | maxDD | IR |
|--|--|--|--|
[填 logs/eval_etf_timing.json]

## 结论与 A2 方向
[单票存活 → A2 薄执行层;ETF 胜出 → ETF 择时转正。诚实记录负结果。]
```

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/specs/2026-06-25-small-capital-execution-diagnostic-results.md
git commit -m "docs(results): A1 small-capital execution diagnostic results"
```

> 注:`logs/*` 与 worktree 内拷贝的 `production/reports/*.pkl` 均被 gitignore,不入库。

---

## Self-Review (已过)

- **Spec 覆盖:** Block1=Task6+Task9.1;Block2=Task3+Task7+Task9.2;Block3=Task5+Task9.4;Block4=Task1+Task4+Task8+Task9.3;对照表+裁决=Task9.5。✓
- **占位扫描:** 仅 `live_trades.csv` 为用户运行时数据(schema 已给),非设计占位;无 TODO/TBD。✓
- **类型一致:** `net_metrics` 键(net_cagr/max_drawdown/net_ir/avg_turnover/cost_drag_annual/win_rate)、`tail_stats` 键(ret_p05/ret_p10/ret_std/neg_period_pct/worst/n)、`build_report(scores, fwd_ret=, policy_name=, top_k=, period=, exit_k=, capital=, profile=, config_path=)`、`compute_exposure(market_close, method=, ma_window=, band=)`、`limit_up_price(instrument, prev_close)` 全部与现有代码核对一致。✓
- **已知取舍(已在文内标注):** tail_stats 用日频 net(非 5 日块);ETF v1 用合成宽基代理;ST ±5% 不特判。
```
