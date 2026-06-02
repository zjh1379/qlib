# P3 风控暴露层（Risk Overlay）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (for CODE tasks T1–T4) or superpowers:executing-plans. T5 is an execution/runbook step. Steps use checkbox (`- [ ]`).

**Goal:** 在现有选股组合之上叠加一层由大盘趋势驱动的 0~1 投资比例（暴露层），把 2 模型组合的 −57% 回撤压到 ~−25%，尽量保住 +19% 收益，零模型重训。

**Architecture:** 三个小单元——`market.py`（universe 等权**尾部**收益→合成大盘代理）、`regime.py`（`compute_exposure` 趋势/波动→[0,1]，无前视）、`engine.py`（目标权重整体×滞后暴露，复用现有成本/换手逻辑）。CLI 用 `run.py --regime` 接入。评估直接打到已有的 `oof_2model_2021_2026.pkl`。

**Tech Stack:** Python, pandas, numpy, qlib, pytest。解释器 `F:/Tools/Anaconda/envs/qlib/python.exe`。仓库根 `E:\Projects\qlib`；pytest 从根目录跑。

---

## 关键约定
- **大盘代理用尾部收益**（`$close/Ref($close,1)-1`，昨日已知），**绝不用** forward `fwd_ret`（含未来）——否则趋势信号前视。
- **无前视**：`compute_exposure` 的 MA/vol 都是尾部窗口；引擎在决策日 `d` 用 `exposure.asof(d)`（≤d 信息，与 `scores(d)` 同口径，次日 open 成交）。
- 复用 P0/P1：`data.init_qlib_from_config`、`engine.run_backtest`、`metrics_net`。

## 文件结构
| 文件 | 职责 |
|---|---|
| `production/backtest/market.py`（新） | 大盘代理：尾部等权收益→累计指数 |
| `production/backtest/regime.py`（新） | `compute_exposure` 暴露信号 |
| `production/backtest/engine.py`（改） | `run_backtest` 加 `exposure` 参数 |
| `production/backtest/run.py`（改） | `--regime` 等 CLI + `build_report` 传 exposure |
| `production/tests/test_market.py`/`test_regime.py`（新）、`test_engine.py`/`test_backtest_run.py`（追加） | 测试 |

---

### Task 1: market.py — 大盘代理

**Files:** Create `production/backtest/market.py`, `production/tests/test_market.py`

- [ ] **Step 1: 写失败测试**
```python
# production/tests/test_market.py
import pandas as pd
import pytest
from production.backtest.market import mean_market_return, returns_to_close, MKT_RET_EXPR


def _ser(d):
    idx = pd.MultiIndex.from_tuples(list(d.keys()), names=["datetime", "instrument"])
    return pd.Series(list(d.values()), index=idx)


def test_mkt_ret_expr_is_trailing():
    # MUST be trailing (Ref +1 = yesterday), never forward (negative Ref)
    assert MKT_RET_EXPR == "$close / Ref($close, 1) - 1"


def test_mean_market_return_equal_weight():
    t1, t2 = pd.Timestamp("2021-01-04"), pd.Timestamp("2021-01-05")
    s = _ser({(t1, "A"): 0.02, (t1, "B"): 0.04, (t2, "A"): -0.01, (t2, "B"): -0.03})
    m = mean_market_return(s)
    assert m.loc[t1] == pytest.approx(0.03)
    assert m.loc[t2] == pytest.approx(-0.02)


def test_returns_to_close_cumprod():
    t1, t2 = pd.Timestamp("2021-01-04"), pd.Timestamp("2021-01-05")
    c = returns_to_close(pd.Series([0.1, -0.1], index=[t1, t2]))
    assert c.loc[t1] == pytest.approx(1.1)
    assert c.loc[t2] == pytest.approx(1.1 * 0.9)
```

- [ ] **Step 2: 运行确认失败**
Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_market.py -v`
Expected: FAIL (`No module named 'production.backtest.market'`)

- [ ] **Step 3: 实现**
```python
# production/backtest/market.py
"""Synthetic broad-market proxy = equal-weight TRAILING daily return of the
universe, cumulated to a close series for the regime/trend signal."""
from __future__ import annotations

# Force installed qlib ahead of the uncompiled ./qlib source tree.
import sys as _sys
import sysconfig as _sysconfig
_PURELIB = _sysconfig.get_paths().get("purelib")
if _PURELIB and _PURELIB not in _sys.path[:1]:
    _sys.path.insert(0, _PURELIB)

import pandas as pd

# Trailing 1-day return (Ref +1 = yesterday). NEVER use forward refs here.
MKT_RET_EXPR = "$close / Ref($close, 1) - 1"


def mean_market_return(ret: pd.Series) -> pd.Series:
    """Equal-weight cross-sectional mean return per datetime."""
    return ret.groupby(level="datetime").mean().sort_index().rename("mkt_ret")


def returns_to_close(mkt_ret: pd.Series) -> pd.Series:
    """Cumulative product -> synthetic market close (base 1.0)."""
    return (1.0 + mkt_ret.fillna(0.0)).cumprod().rename("market_close")


def load_market_proxy(instruments, start: str, end: str,
                      config_path: str = "production/configs/rolling_ensemble.yaml") -> pd.Series:
    """Load trailing 1d returns for `instruments`, equal-weight mean per day,
    cumulate to a synthetic market close Series indexed by datetime."""
    from qlib.data.dataset.loader import QlibDataLoader
    from .data import init_qlib_from_config
    init_qlib_from_config(config_path)
    loader = QlibDataLoader(config={"feature": ([MKT_RET_EXPR], ["mkt_ret"])})
    df = loader.load(instruments=instruments, start_time=start, end_time=end)
    s = df.iloc[:, 0] if isinstance(df, pd.DataFrame) else df
    if s.index.names[0] == "instrument":
        s = s.swaplevel().sort_index()
    s.index = s.index.set_names(["datetime", "instrument"])
    return returns_to_close(mean_market_return(s.dropna()))
```

- [ ] **Step 4: 运行确认通过**
Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_market.py -v`
Expected: 3 passed

- [ ] **Step 5: 提交**
```bash
git add production/backtest/market.py production/tests/test_market.py
git commit -m "feat(risk): market proxy (equal-weight trailing returns -> close)"
```

---

### Task 2: regime.py — 暴露信号

**Files:** Create `production/backtest/regime.py`, `production/tests/test_regime.py`

- [ ] **Step 1: 写失败测试**
```python
# production/tests/test_regime.py
import numpy as np
import pandas as pd
import pytest
from production.backtest.regime import compute_exposure


def _rising(n=200):
    return pd.Series(np.arange(1, n + 1, dtype=float), index=pd.bdate_range("2020-01-01", periods=n))


def _falling(n=200):
    return pd.Series(np.arange(n, 0, -1, dtype=float), index=pd.bdate_range("2020-01-01", periods=n))


def test_uptrend_full_exposure():
    e = compute_exposure(_rising(), ma_window=20, band=0.10)
    assert e.iloc[-1] == pytest.approx(1.0)


def test_downtrend_floor_zero():
    e = compute_exposure(_falling(), ma_window=20, band=0.10, min_exposure=0.0)
    assert e.iloc[-1] == pytest.approx(0.0)


def test_warmup_is_full_exposure():
    e = compute_exposure(_rising(10), ma_window=20)
    assert (e == 1.0).all()


def test_no_lookahead_changing_future_doesnt_change_past():
    base = pd.Series([10.0] * 60 + [11, 12, 13, 14, 15],
                     index=pd.bdate_range("2020-01-01", periods=65), dtype=float)
    e1 = compute_exposure(base, ma_window=20, band=0.10)
    mod = base.copy(); mod.iloc[-1] = 999.0
    e2 = compute_exposure(mod, ma_window=20, band=0.10)
    assert e1.iloc[:-1].equals(e2.iloc[:-1])


def test_vol_target_compresses_exposure():
    # flat-but-jumpy series above its MA: trend ~full, but high vol -> vol_target cuts it
    rng = np.random.default_rng(0)
    px = pd.Series(100 + np.cumsum(rng.normal(0, 3, 200)),
                   index=pd.bdate_range("2020-01-01", periods=200))
    e_trend = compute_exposure(px, ma_window=20, band=0.10)
    e_vol = compute_exposure(px, ma_window=20, band=0.10, vol_target=0.05, vol_window=20)
    assert e_vol.iloc[-1] <= e_trend.iloc[-1] + 1e-9
```

- [ ] **Step 2: 运行确认失败**
Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_regime.py -v`
Expected: FAIL (`No module named 'production.backtest.regime'`)

- [ ] **Step 3: 实现**
```python
# production/backtest/regime.py
"""Market-regime exposure signal in [0,1]. Trailing-only (no lookahead)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_exposure(market_close: pd.Series, method: str = "trend_ma",
                     ma_window: int = 120, band: float = 0.10,
                     min_exposure: float = 0.0, vol_target: float | None = None,
                     vol_window: int = 20, periods_per_year: int = 252) -> pd.Series:
    """Daily exposure in [0,1] from a synthetic market close.
    trend_ma: e = clip((close/MA - 1)/band + 0.5, min_exposure, 1); warm-up -> 1.0.
    Optional vol_target overlay: e *= min(1, vol_target/realized_vol); may push below floor.
    """
    if method != "trend_ma":
        raise ValueError(f"unknown method {method!r}")
    close = market_close.sort_index()
    ma = close.rolling(ma_window, min_periods=ma_window).mean()
    raw = (close / ma - 1.0) / band + 0.5
    e = raw.clip(lower=min_exposure, upper=1.0)
    e = e.where(ma.notna(), 1.0)  # warm-up: stay fully invested
    if vol_target is not None:
        ret = close.pct_change()
        rv = ret.rolling(vol_window, min_periods=vol_window).std() * np.sqrt(periods_per_year)
        vt = (vol_target / rv).clip(upper=1.0)
        vt = vt.where(rv.notna(), 1.0)
        e = (e * vt).clip(lower=0.0, upper=1.0)  # vol can push to full risk-off
    return e.rename("exposure")
```

- [ ] **Step 4: 运行确认通过**
Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_regime.py -v`
Expected: 5 passed

- [ ] **Step 5: 提交**
```bash
git add production/backtest/regime.py production/tests/test_regime.py
git commit -m "feat(risk): regime exposure signal (trend MA + optional vol target)"
```

---

### Task 3: engine.py — 暴露应用

**Files:** Modify `production/backtest/engine.py`; Test append `production/tests/test_engine.py`

- [ ] **Step 1: 追加失败测试**
```python
# append to production/tests/test_engine.py
def test_constant_exposure_scales_gross_linearly():
    dates = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    scores = _series({(dates[0], "A"): 1.0, (dates[1], "A"): 1.0})
    fwd = _series({(dates[0], "A"): 0.01, (dates[1], "A"): 0.02})
    full = run_backtest(scores, fwd, Daily(top_k=1), CostModel(), 1e5)["daily"]
    exp = pd.Series(0.5, index=pd.Index(dates, name="datetime"))
    half = run_backtest(scores, fwd, Daily(top_k=1), CostModel(), 1e5, exposure=exp)["daily"]
    # second day holds same name -> no cost; gross should be exactly half
    assert half.iloc[1]["gross"] == pytest.approx(0.5 * full.iloc[1]["gross"], rel=1e-9)


def test_zero_exposure_no_gross():
    dates = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    scores = _series({(dates[0], "A"): 1.0, (dates[1], "A"): 1.0})
    fwd = _series({(dates[0], "A"): 0.05, (dates[1], "A"): 0.05})
    exp = pd.Series(0.0, index=pd.Index(dates, name="datetime"))
    res = run_backtest(scores, fwd, Daily(top_k=1), CostModel(), 1e5, exposure=exp)["daily"]
    assert res["gross"].abs().sum() == pytest.approx(0.0)


def test_exposure_reduces_drawdown_on_bear():
    import numpy as np
    dates = pd.bdate_range("2024-01-02", periods=60)
    stocks = ["A", "B"]
    idx = pd.MultiIndex.from_product([dates, stocks], names=["datetime", "instrument"])
    scores = pd.Series(1.0, index=idx)
    # bear: -1%/day for second half
    rets = {d: (0.005 if i < 30 else -0.02) for i, d in enumerate(dates)}
    fwd = pd.Series([rets[d] for (d, _) in idx], index=idx)
    base = run_backtest(scores, fwd, Daily(top_k=2), CostModel(), 1e5)["daily"]
    # exposure: full first half, 0 in bear
    exp = pd.Series([1.0 if i < 30 else 0.0 for i in range(len(dates))],
                    index=pd.Index(dates, name="datetime"))
    over = run_backtest(scores, fwd, Daily(top_k=2), CostModel(), 1e5, exposure=exp)["daily"]
    base_dd = ((1 + base["net"]).cumprod() / (1 + base["net"]).cumprod().cummax() - 1).min()
    over_dd = ((1 + over["net"]).cumprod() / (1 + over["net"]).cumprod().cummax() - 1).min()
    assert over_dd > base_dd  # overlay drawdown is less negative
```

- [ ] **Step 2: 运行确认失败**
Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_engine.py -k "exposure" -v`
Expected: FAIL (`run_backtest() got an unexpected keyword argument 'exposure'`)

- [ ] **Step 3: 实现** — modify `run_backtest` in `production/backtest/engine.py`

Change the signature line:
```python
def run_backtest(
    scores,
    fwd_ret: pd.Series,
    policy: RebalancePolicy,
    cost: CostModel,
    capital: float = 100_000.0,
    score_col: str = "score",
    exposure: pd.Series | None = None,
) -> dict:
```
Inside the loop, immediately AFTER `target = policy.target_weights(s_d, current)` and BEFORE `allidx = ...`, insert:
```python
            if exposure is not None and len(target):
                e = exposure.asof(d) if len(exposure) and d >= exposure.index[0] else 1.0
                if pd.isna(e):
                    e = 1.0
                e = float(min(1.0, max(0.0, e)))
                target = target * e
```
(The existing delta/turnover/cost and `gross = Σ(current·fwd_ret)` logic then naturally handles the scaled weights — exposure changes show up as turnover, and gross = e·portfolio return.)

- [ ] **Step 4: 运行确认通过**
Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_engine.py -v`
Expected: all pass (3 original + 3 new = 6)

- [ ] **Step 5: 提交**
```bash
git add production/backtest/engine.py production/tests/test_engine.py
git commit -m "feat(risk): engine exposure overlay (scale target weights, no lookahead)"
```

---

### Task 4: run.py — `--regime` 接入

**Files:** Modify `production/backtest/run.py`; Test append `production/tests/test_backtest_run.py`

- [ ] **Step 1: 追加失败测试**
```python
# append to production/tests/test_backtest_run.py
def test_build_report_accepts_exposure():
    dates = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    scores = _series({(dates[0], "A"): 1.0, (dates[1], "A"): 1.0})
    fwd = _series({(dates[0], "A"): 0.05, (dates[1], "A"): 0.05})
    exp = pd.Series(0.0, index=pd.Index(dates, name="datetime"))
    rep = build_report(scores, fwd, policy_name="daily", top_k=1, period=5, exit_k=2,
                       capital=100_000.0, profile="small", exposure=exp)
    # zero exposure -> ~zero net return -> nav unchanged
    assert rep["final_nav"] == pytest.approx(100_000.0, rel=1e-6)
```
(Top of file already has `_series` from the earlier engine-style helper? If not present in this file, add:
```python
def _series(d):
    idx = pd.MultiIndex.from_tuples(list(d.keys()), names=["datetime", "instrument"])
    return pd.Series(list(d.values()), index=idx)
```
near the top of `test_backtest_run.py`.)

- [ ] **Step 2: 运行确认失败**
Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_backtest_run.py -k exposure -v`
Expected: FAIL (`build_report() got an unexpected keyword argument 'exposure'`)

- [ ] **Step 3: 实现** — modify `production/backtest/run.py`

(a) `build_report` — add `exposure=None` param and pass through:
```python
def build_report(scores: pd.Series, fwd_ret: pd.Series, *, policy_name: str,
                 top_k: int, period: int, exit_k: int, capital: float,
                 profile: str, exposure=None) -> dict:
    policy = _make_policy(policy_name, top_k, period, exit_k)
    cm = cost_model(profile)
    res = run_backtest(scores, fwd_ret, policy, cm, capital=capital, exposure=exposure)
    return {
        "params": {"policy": policy_name, "top_k": top_k, "period": period,
                   "exit_k": exit_k, "capital": capital, "profile": profile,
                   "regime": getattr(exposure, "name", None) if exposure is not None else None},
        "metrics": net_metrics(res["daily"]),
        "regimes": net_regime(res["daily"]),
        "final_nav": res["final_nav"],
        "generated_at": datetime.utcnow().isoformat(),
    }
```

(b) `main()` — add args (next to the others):
```python
    ap.add_argument("--regime", default="none", choices=["none", "trend_ma"])
    ap.add_argument("--ma-window", type=int, default=120)
    ap.add_argument("--band", type=float, default=0.10)
    ap.add_argument("--min-exposure", type=float, default=0.0)
    ap.add_argument("--vol-target", type=float, default=None)
```
and AFTER `fwd = load_fwd_returns(...)`, BEFORE `build_report(...)`, insert:
```python
    exposure = None
    if args.regime != "none":
        from .market import load_market_proxy
        from .regime import compute_exposure
        mkt = load_market_proxy(instruments, start, end, config_path=args.config)
        exposure = compute_exposure(mkt, method=args.regime, ma_window=args.ma_window,
                                    band=args.band, min_exposure=args.min_exposure,
                                    vol_target=args.vol_target)
```
and pass `exposure=exposure` into the `build_report(...)` call.

- [ ] **Step 4: 运行确认通过**
Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_backtest_run.py -v`
Expected: all pass

- [ ] **Step 5: 提交**
```bash
git add production/backtest/run.py production/tests/test_backtest_run.py
git commit -m "feat(risk): wire --regime exposure overlay into run CLI"
```

---

### Task 5（执行）: 调参命中 −25% 回撤 + 结果

- [ ] **基线对照**（无 overlay，已存在则复用）：`... -m production.backtest.run --pred-file production/reports/oof_2model_2021_2026.pkl --policy fixed --top-k 5 --period 5 --capital 100000 --profile small --out production/reports/risk_base.json`
- [ ] **扫 regime 参数**：对 `ma_window ∈ {60,120,200}` × `min_exposure ∈ {0.0,0.2}` ×（可选 `vol_target ∈ {none,0.15}`）跑 `run --regime trend_ma --ma-window M --min-exposure X [--vol-target V] --out production/reports/risk_ma{M}_min{X}[_vt{V}].json`（一个小 shell 循环）。
- [ ] **挑配置**：用一个 python 片段读所有 `risk_*.json`，打印每个的 `max_drawdown / net_cagr / net_ir / Calmar(=net_cagr/abs(maxDD))`，选 **maxDD∈[−30%,−20%] 且 net_cagr 最高**的配置。
- [ ] **验收**（§9）：选中配置 maxDD≈−25%、net_cagr>+10%目标、各 regime 净IR为正、Calmar 明显优于基线(~0.33)。
- [ ] **结果文档 + 回归 + 提交**：写 `docs/superpowers/specs/2026-06-02-risk-overlay-results.md`（基线 vs 最优 overlay：回撤/收益/Calmar/各regime/暴露曲线特征）；跑全 backtest 套件（含 test_market/test_regime/test_engine/test_backtest_run）确认无回归；`git add` 结果 json + 文档并提交 `docs(p3): risk-overlay results (DD ~-25%, Calmar improved)`。

---

## Self-Review
**Spec 覆盖**：§4 market.py→T1；§5 regime.py→T2；§6 引擎集成→T3；§4-7 CLI/数据流→T4；§9 验收→T5。✅
**无前视**：MKT_RET_EXPR 钉死为尾部（T1 测试）；compute_exposure no-lookahead 测试（T2）；引擎 `exposure.asof(d)`（T3）。✅
**Placeholder**：无；每步含完整代码与命令。T5 为执行 runbook（参数网格小、命令明确）。
**类型一致**：`mean_market_return`/`returns_to_close`/`load_market_proxy`（T1）、`compute_exposure(market_close, method, ma_window, band, min_exposure, vol_target, vol_window)`（T2）、`run_backtest(..., exposure=None)`（T3）、`build_report(..., exposure=None)`（T4）签名前后一致；引擎 `exposure.asof(d)` 依赖 market_close/exposure 按 datetime 排序（compute_exposure 已 sort）。✅
**风险**：T3 改 run_backtest 签名——`exposure` 末位带默认 None，向后兼容（现有调用不传即原行为）；现有 test_engine 原 3 测试不受影响。

## 后续
- 可选真大盘指数对照；个股止损/行业上限（P3.1）；把"持5只/5天换+暴露层"落进产品。
