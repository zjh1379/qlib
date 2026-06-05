# 日内执行择时 P1 (离线模拟器) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 离线检验"把 factor-2model top-5 的进场价从次日开盘换成日内择时价"能否提升扣费后净收益,零重训、用免费 baostock 5min。

**Architecture:** 三个小组件——`fetch_5min`(按需拉 5min+缓存)、`entry_rules`(纯函数:从一天 5min 算"相对开盘的进场乘数",含 A 股涨停不可成交判定)、`exec_backtest`(从 OOF 取 picks→映射进/出场日→把乘数乘到日线复权开盘价→聚合成周期净收益,复用 `metrics_net`)。**关键:进场规则输出"乘数 = 规则价/当日开盘"(同日比值,复权无关)**,模拟器再乘到引擎的复权开盘价,彻底回避复权口径问题。

**Tech Stack:** Python, baostock(5min), pandas, qlib(QlibDataLoader $open + D.calendar), pytest。解释器 `F:/Tools/Anaconda/envs/qlib/python.exe`;**所有命令在主仓库 `E:\Projects\qlib` 跑,用 `-m`、加 `-X utf8`**;改 production 代码走 feature 分支。

---

## File Structure
| 文件 | 职责 |
|---|---|
| `production/intraday/__init__.py` | 空包标记 |
| `production/intraday/entry_rules.py` | 纯函数:`entry_multiplier(day_bars, prev_close, instrument, rule, **p)`、`is_buy_fillable`、`limit_up_price`、`bs_code`(SH600→sh.600) |
| `production/intraday/fetch_5min.py` | baostock 拉 5min + 当日/前日 daily(取 raw prev_close)+ parquet 缓存 |
| `production/intraday/exec_backtest.py` | 选股(FixedPeriod top-5)→进/出场日→乘数×复权开盘→周期净收益→`metrics_net` 对比各规则 |
| `production/tests/test_entry_rules.py` | TDD:乘数/VWAP/低点带/gap/首30/涨停不可成交 |
| `production/tests/test_exec_backtest.py` | TDD:选股+进出场映射+聚合(合成数据) |
| `production/intraday/cache/` | parquet 缓存(加进 .gitignore) |
| `docs/superpowers/specs/2026-06-04-intraday-execution-results.md` | P1 结果(跑完写) |

---

## Task 0（runbook·数据 spike·阻断门）: 确认 baostock 5min 可用

- [ ] **Step 1: 实测拉一只票几天 5min**
Run:
```
F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -c "import baostock as bs; bs.login(); rs=bs.query_history_k_data_plus('sh.600519','time,open,high,low,close,volume,amount','2024-12-02','2024-12-06',frequency='5',adjustflag='3'); df=rs.get_data(); bs.logout(); print(df.shape); print(df.head()); print(df.tail(2))"
```
Expected: 非空 DataFrame,每天 48 根 5min(09:30–11:30 + 13:00–15:00),`time` 形如 `20241202093500000`,价为字符串数字。
- [ ] **Step 2: 记录字段/时区/复权事实** 到 `docs/.../2026-06-04-intraday-execution-results.md` 顶部"数据可得性"一节(根数、time 解析、是否含 amount、raw 与日线 $open 的同日比值是否≈1)。
- [ ] **Step 3: 阻断判定** 若拉不到/字段缺失 → 停止,改用 akshare `stock_zh_a_hist_min_em` 重做本 Task 再继续(更新 fetch 设计)。拉得到 → 继续 Task 1。

---

## Task 1: `entry_rules.py`（纯函数,TDD）

**Files:** Create `production/intraday/entry_rules.py`; Test `production/tests/test_entry_rules.py`

- [ ] **Step 1: 写失败测试**
```python
# production/tests/test_entry_rules.py
import numpy as np, pandas as pd, pytest
from production.intraday.entry_rules import (
    bs_code, limit_up_price, is_buy_fillable, entry_multiplier)

def _day(opens, highs, lows, closes, vols):  # one day's 5min bars
    n=len(opens)
    return pd.DataFrame({"open":opens,"high":highs,"low":lows,"close":closes,
                         "volume":vols,"amount":np.array(closes)*np.array(vols)})

def test_bs_code():
    assert bs_code("SH600519")=="sh.600519"
    assert bs_code("SZ000001")=="sz.000001"

def test_limit_up_price_by_board():
    assert limit_up_price("SH600000",10.0)==pytest.approx(11.0)   # main 10%
    assert limit_up_price("SZ300750",10.0)==pytest.approx(12.0)   # ChiNext 20%
    assert limit_up_price("SH688981",10.0)==pytest.approx(12.0)   # STAR 20%
    assert limit_up_price("BJ830799",10.0)==pytest.approx(13.0)   # BJ 30%

def test_open_multiplier_is_one():
    d=_day([10,10.1],[10.2,10.2],[9.9,10.0],[10.1,10.1],[100,100])
    assert entry_multiplier(d,prev_close=9.8,instrument="SH600000",rule="open")==pytest.approx(1.0)

def test_low_band_filled_when_touched():
    # open 10, dips to 9.8 (=-2%): low_band(0.01)=9.9 touched -> entry 9.9 -> mult 0.99
    d=_day([10,9.8],[10.0,9.9],[10.0,9.8],[10.0,9.85],[100,100])
    m=entry_multiplier(d,9.9,"SH600000",rule="low_band",k=0.01)
    assert m==pytest.approx(0.99)

def test_low_band_missed_uses_close():
    # never dips to band -> fill at day close
    d=_day([10,10.1],[10.2,10.2],[10.0,10.05],[10.1,10.15],[100,100])
    m=entry_multiplier(d,9.9,"SH600000",rule="low_band",k=0.02)  # band 9.8 not touched
    assert m==pytest.approx(10.15/10.0)

def test_vwap_multiplier():
    d=_day([10,10],[10,10],[10,10],[10,12],[100,100])  # closes 10,12 vol 100,100
    # vwap = (10*100+12*100)/200 = 11 ; open=10 -> mult 1.1
    assert entry_multiplier(d,9.9,"SH600000",rule="vwap")==pytest.approx(1.1)

def test_gap_cond_skips_big_gap_up():
    # open 10, prev_close 9.0 -> gap +11% >= g(3%) -> not chasing -> None
    d=_day([10,10.1],[10.2,10.2],[9.95,10.0],[10.1,10.1],[100,100])
    assert entry_multiplier(d,9.0,"SH600000",rule="gap_cond",g=0.03) is None

def test_gap_cond_buys_on_low_open():
    d=_day([10,10.1],[10.2,10.2],[9.95,10.0],[10.1,10.1],[100,100])
    assert entry_multiplier(d,10.5,"SH600000",rule="gap_cond",g=0.03)==pytest.approx(1.0)  # gap down -> open

def test_first30_low_uses_min_first6_bars():
    opens=[10,10,10,10,10,10,10]; lows=[10,9.9,9.8,9.95,10,10,10]
    d=_day(opens,[11]*7,lows,[10]*7,[100]*7)
    # min low over first 6 bars = 9.8 -> mult 0.98
    assert entry_multiplier(d,9.9,"SH600000",rule="first30_low")==pytest.approx(0.98)

def test_one_zi_limit_up_not_fillable():
    # 一字涨停: whole day at limit (open=high=low=close=11.0=prev*1.1) -> buy unfillable
    d=_day([11.0,11.0],[11.0,11.0],[11.0,11.0],[11.0,11.0],[100,100])
    assert is_buy_fillable(d,prev_close=10.0,instrument="SH600000") is False
    assert entry_multiplier(d,10.0,"SH600000",rule="open") is None
```
- [ ] **Step 2: 跑测试确认失败**
Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_entry_rules.py -q` → ImportError。
- [ ] **Step 3: 写实现**
```python
# production/intraday/entry_rules.py
"""Intraday entry-timing rules. Each rule returns a MULTIPLIER m relative to the
day's open (effective_entry = day_open * m), so it is 复权-invariant (same-day
ratio). Returns None when a BUY is not fillable (A-share limit-up). The simulator
multiplies m onto the daily ADJUSTED open."""
from __future__ import annotations
import pandas as pd


def bs_code(instrument: str) -> str:
    """SH600519 -> sh.600519 ; SZ000001 -> sz.000001 (baostock format)."""
    return f"{instrument[:2].lower()}.{instrument[2:]}"


def _limit_pct(instrument: str) -> float:
    code = instrument[2:]
    if instrument.startswith("BJ") or code.startswith(("43", "83", "87", "88", "92")):
        return 0.30                       # 北交所
    if code.startswith("688"):
        return 0.20                       # 科创板
    if code.startswith("30"):
        return 0.20                       # 创业板
    return 0.10                           # 主板


def limit_up_price(instrument: str, prev_close: float) -> float:
    return round(prev_close * (1 + _limit_pct(instrument)), 2)


def is_buy_fillable(day_bars: pd.DataFrame, prev_close: float, instrument: str) -> bool:
    """Not fillable if the stock never trades below its limit-up price (一字/封板涨停):
    a buyer can't get filled below the ceiling all day."""
    if day_bars is None or day_bars.empty:
        return False
    lu = limit_up_price(instrument, prev_close)
    return float(day_bars["low"].min()) < lu - 1e-9


def entry_multiplier(day_bars: pd.DataFrame, prev_close: float, instrument: str,
                     rule: str = "open", *, k: float = 0.01, g: float = 0.03,
                     first_n: int = 6) -> float | None:
    if day_bars is None or day_bars.empty:
        return None
    if not is_buy_fillable(day_bars, prev_close, instrument):
        return None
    o = float(day_bars["open"].iloc[0])
    if o <= 0:
        return None
    close = float(day_bars["close"].iloc[-1])
    if rule == "open":
        price = o
    elif rule == "vwap":
        vol = float(day_bars["volume"].sum())
        price = float(day_bars["amount"].sum()) / vol if vol > 0 else o
    elif rule == "vwap_am":
        am = day_bars[day_bars.index < len(day_bars) // 2] if len(day_bars) else day_bars
        vol = float(am["volume"].sum())
        price = float(am["amount"].sum()) / vol if vol > 0 else o
    elif rule == "low_band":
        band = o * (1 - k)
        price = band if float(day_bars["low"].min()) <= band else close
    elif rule == "gap_cond":
        gap = o / prev_close - 1 if prev_close > 0 else 0.0
        if gap >= g:
            return None                   # don't chase a gap-up
        price = o
    elif rule == "first30_low":
        price = float(day_bars["low"].iloc[:first_n].min())
    else:
        raise ValueError(f"unknown rule {rule!r}")
    return price / o
```
- [ ] **Step 4: 跑测试通过**
Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_entry_rules.py -q` → all pass。
- [ ] **Step 5: 提交**
```
git add production/intraday/__init__.py production/intraday/entry_rules.py production/tests/test_entry_rules.py
git commit -m "feat(intraday): entry-timing rules (multiplier vs open, limit-up aware)"
```
(先 `New-Item production/intraday/__init__.py`(空)。)

---

## Task 2: `fetch_5min.py`（baostock 拉取 + 缓存）

**Files:** Create `production/intraday/fetch_5min.py`; Test append `production/tests/test_entry_rules.py`(纯函数 `parse_baostock_5min`)

- [ ] **Step 1: 写失败测试(只测纯解析,不联网)**
```python
# append to test_entry_rules.py
def test_parse_baostock_5min_types():
    from production.intraday.fetch_5min import parse_baostock_5min
    raw = pd.DataFrame({
        "time":["20241202093500000","20241202094000000"],
        "open":["10.0","10.1"],"high":["10.2","10.2"],"low":["9.9","10.0"],
        "close":["10.1","10.15"],"volume":["100","200"],"amount":["1010","2030"]})
    out = parse_baostock_5min(raw)
    assert str(out["datetime"].iloc[0])=="2024-12-02 09:35:00"
    assert out["open"].dtype.kind=="f" and out["volume"].iloc[1]==200.0
    assert list(out.columns)[:1]==["datetime"]
```
- [ ] **Step 2: 跑确认失败** → ImportError。
- [ ] **Step 3: 写实现**
```python
# production/intraday/fetch_5min.py
"""Fetch raw 5-min bars (+ prev daily close) from baostock, with parquet cache.
Network calls are isolated here; entry_rules/exec_backtest stay pure/offline."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE = REPO_ROOT / "production" / "intraday" / "cache"


def parse_baostock_5min(raw: pd.DataFrame) -> pd.DataFrame:
    """baostock 5min get_data() (all-string cols) -> typed, with parsed `datetime`."""
    df = pd.DataFrame()
    df["datetime"] = pd.to_datetime(raw["time"].str[:14], format="%Y%m%d%H%M%S")
    for c in ("open", "high", "low", "close", "volume", "amount"):
        df[c] = pd.to_numeric(raw[c], errors="coerce")
    return df[["datetime", "open", "high", "low", "close", "volume", "amount"]]


def fetch_5min(instrument: str, start: str, end: str) -> pd.DataFrame:
    """Return typed 5min bars for [start,end] inclusive. Cached per inst+month is
    overkill for P1; cache per (inst,start,end) call as one parquet."""
    from production.intraday.entry_rules import bs_code
    CACHE.mkdir(parents=True, exist_ok=True)
    fp = CACHE / f"{instrument}_{start}_{end}.parquet"
    if fp.exists():
        return pd.read_parquet(fp)
    import baostock as bs
    bs.login()
    try:
        rs = bs.query_history_k_data_plus(
            bs_code(instrument), "time,open,high,low,close,volume,amount",
            start_date=start, end_date=end, frequency="5", adjustflag="3")
        raw = rs.get_data()
    finally:
        bs.logout()
    out = parse_baostock_5min(raw) if raw is not None and len(raw) else \
        pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume", "amount"])
    out.to_parquet(fp)
    return out


def prev_close_raw(instrument: str, date: str) -> float | None:
    """Raw (unadjusted) close of the last trading day strictly before `date`,
    via baostock daily — needed for limit-up / gap detection on raw intraday."""
    from production.intraday.entry_rules import bs_code
    start = (pd.Timestamp(date) - pd.Timedelta(days=12)).date().isoformat()
    import baostock as bs
    bs.login()
    try:
        rs = bs.query_history_k_data_plus(bs_code(instrument), "date,close",
                                          start_date=start, end_date=date,
                                          frequency="d", adjustflag="3")
        d = rs.get_data()
    finally:
        bs.logout()
    if d is None or len(d) < 2:
        return None
    d = d[d["date"] < date]
    return float(d["close"].iloc[-1]) if len(d) else None
```
- [ ] **Step 4: 跑解析测试通过** `pytest production/tests/test_entry_rules.py -q`。
- [ ] **Step 5: 提交** `git add production/intraday/fetch_5min.py production/tests/test_entry_rules.py; git commit -m "feat(intraday): baostock 5min fetch + parse + cache"`

---

## Task 3: `exec_backtest.py`（模拟器,TDD 选股+聚合）

**Files:** Create `production/intraday/exec_backtest.py`; Test `production/tests/test_exec_backtest.py`

- [ ] **Step 1: 写失败测试(选股+进出场映射,合成数据,不联网)**
```python
# production/tests/test_exec_backtest.py
import pandas as pd, pytest
from production.intraday.exec_backtest import enumerate_trades

def _scores():
    days = pd.date_range("2024-01-01", periods=12, freq="B")
    rows=[]
    for d in days:
        for i,inst in enumerate(["A","B","C"]):
            rows.append(((d,inst), 3-i))  # A>B>C every day
    s=pd.Series(dict(rows)); s.index=s.index.set_names(["datetime","instrument"]); return s

def test_enumerate_trades_fixed_5d_topk2():
    s=_scores()
    trades=enumerate_trades(s, top_k=2, period=5)
    # rebalance at step 0 and 5; entry = NEXT trading day's open; top-2 = A,B
    assert {t["instrument"] for t in trades} == {"A","B"}
    t0=[t for t in trades if t["rebalance_step"]==0]
    assert len(t0)==2
    # entry_date is the trading day AFTER the decision day; exit_date 5 sessions later
    assert t0[0]["entry_date"] > t0[0]["decision_date"]
    assert t0[0]["exit_date"] > t0[0]["entry_date"]
```
- [ ] **Step 2: 跑确认失败** → ImportError。
- [ ] **Step 3: 写实现(选股+映射;打分用 FixedPeriod 口径)**
```python
# production/intraday/exec_backtest.py
"""Offline intraday-execution simulator: replace next-open entry with rule-based
intraday entry for the factor-2model fixed/hold-5/5d top-k picks; compare net."""
from __future__ import annotations
import sys, sysconfig
_P = sysconfig.get_paths().get("purelib")
if _P and _P not in sys.path[:1]:
    sys.path.insert(0, _P)
from pathlib import Path
import numpy as np, pandas as pd


def enumerate_trades(scores: pd.Series, top_k: int = 5, period: int = 5) -> list[dict]:
    """Walk the fixed/period rebalance schedule; on each rebalance day pick top_k
    by score; map decision_date -> entry_date (next session) -> exit_date (+period
    sessions). Returns one dict per (rebalance, name)."""
    dates = sorted(scores.index.get_level_values("datetime").unique())
    out = []
    for step, i in enumerate(range(0, len(dates), period)):
        d = dates[i]
        if i + 1 >= len(dates):
            break
        entry = dates[i + 1]
        exit_i = min(i + 1 + period, len(dates) - 1)
        exit_ = dates[exit_i]
        cross = scores.xs(d, level="datetime").dropna().sort_values(ascending=False)
        for inst in list(cross.index[:top_k]):
            out.append({"rebalance_step": i, "decision_date": d, "entry_date": entry,
                        "exit_date": exit_, "instrument": inst})
    return out
```
- [ ] **Step 4: 跑测试通过** `pytest production/tests/test_exec_backtest.py -q`。
- [ ] **Step 5: 加"应用规则+聚合"函数(集成,无单测,Task5 跑真数据验证)**
在 `exec_backtest.py` 追加:
```python
def daily_open_adj(instruments, start, end, config="production/configs/rolling_ensemble.yaml") -> pd.Series:
    """Adjusted daily $open per (datetime,instrument) via qlib (engine-consistent)."""
    from qlib.data.dataset.loader import QlibDataLoader
    from production.backtest.data import init_qlib_from_config
    init_qlib_from_config(config)
    px = QlibDataLoader(config={"feature": (["$open"], ["open"])}).load(
        instruments=instruments, start_time=start, end_time=end)
    s = px.iloc[:, 0] if isinstance(px, pd.DataFrame) else px
    if s.index.names[0] == "instrument":
        s = s.swaplevel().sort_index()
    s.index = s.index.set_names(["datetime", "instrument"]); return s.sort_index()


def simulate(scores, *, rule, top_k=5, period=5, k=0.01, g=0.03,
             cost_bps=10.0) -> dict:
    """For each trade: entry_adj = open_adj(entry) * entry_multiplier(rule);
    ret = open_adj(exit)/entry_adj - 1 - cost; aggregate equal-weight per
    rebalance into a period-return series -> net metrics. rule='open' reproduces
    the open baseline (multiplier 1.0, no fetch)."""
    from production.intraday.entry_rules import entry_multiplier
    from production.intraday.fetch_5min import fetch_5min, prev_close_raw
    trades = enumerate_trades(scores, top_k, period)
    insts = sorted({t["instrument"] for t in trades})
    dmin = min(t["entry_date"] for t in trades); dmax = max(t["exit_date"] for t in trades)
    opens = daily_open_adj(insts, str(dmin.date()), str(dmax.date()))
    per_rebalance: dict = {}
    n_skip = n_fallback = 0
    for t in trades:
        oe = opens.get((t["entry_date"], t["instrument"]))
        ox = opens.get((t["exit_date"], t["instrument"]))
        if oe is None or ox is None or not (oe > 0):
            continue
        mult = 1.0
        if rule != "open":
            ed = t["entry_date"].strftime("%Y-%m-%d")
            bars = fetch_5min(t["instrument"], ed, ed)
            pc = prev_close_raw(t["instrument"], ed)
            m = entry_multiplier(bars, pc if pc else 0.0, t["instrument"],
                                 rule=rule, k=k, g=g) if pc else None
            if m is None:
                n_skip += 1; continue       # unfillable / don't-chase -> skip trade
            mult = m
        entry_adj = float(oe) * mult
        ret = float(ox) / entry_adj - 1 - cost_bps / 1e4
        per_rebalance.setdefault(t["rebalance_step"], []).append(ret)
    periods = sorted(per_rebalance)
    pr = pd.Series([np.mean(per_rebalance[p]) for p in periods],
                   index=[scores.index.get_level_values("datetime").unique()[0]] * 0 + periods)
    eq = (1 + pr).cumprod()
    n = len(pr)
    ann = (eq.iloc[-1] ** (252 / (period * n)) - 1) if n and eq.iloc[-1] > 0 else float("nan")
    dd = float((eq / eq.cummax() - 1).min()) if n else float("nan")
    return {"rule": rule, "net_cagr": ann, "calmar": (ann / abs(dd)) if dd else float("nan"),
            "max_dd": dd, "win": float((pr > 0).mean()) if n else float("nan"),
            "n_periods": n, "n_skipped": n_skip}
```
- [ ] **Step 6: 提交** `git add production/intraday/exec_backtest.py production/tests/test_exec_backtest.py; git commit -m "feat(intraday): execution simulator (trade enumeration + rule apply + net)"`

---

## Task 4（runbook·拉真数据）: 预热 5min 缓存

- [ ] **Step 1: 写一次性脚本 `production/_run_intraday_p1.py`**(`-m` 运行):载入 `oof_lgbmfac`+`oof_2model` 重建 factor-2model score(用 `_eval_factors._rebuild_2model`)→ `simulate(rule='open')` 先跑通(只用日线、不拉 5min,作回归锚)→ 打印 baseline。
- [ ] **Step 2: 跑 baseline 锚**
Run: `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m production._run_intraday_p1 --rule open > logs/intraday_open.log 2>&1`
Expected: net_cagr 与现引擎 fixed/5/5 的 +31% **同量级**(口径近似,允许小差;若差很大先查 enumerate/对齐)。
- [ ] **Step 3: 预热缓存**(rule 非 open 会触发 fetch;先小窗验证联网拉取 + 缓存命中)。在 1 个最近调仓期上跑 `rule=vwap`,确认 `production/intraday/cache/*.parquet` 落盘、第二次跑命中缓存秒回、`n_skipped` 合理。

---

## Task 5（runbook·扫描+出结论+存档）

- [ ] **Step 1: 全规则扫描** 跑 `open / vwap / vwap_am / low_band(k=0.005,0.01,0.02) / gap_cond(g=0.02,0.03) / first30_low`,各出 net_cagr/Calmar/maxDD/win/n_skipped + 进场价改善中位 bp + 不可成交占比 → `logs/intraday_sweep.log` + `_summary.json`。
- [ ] **Step 2: 写结果 spec** `docs/superpowers/specs/2026-06-04-intraday-execution-results.md`:对比表(各规则 vs open 基线)、逐年(尤其 2022/2023)、与反转理论一致性、不可成交/回退占比、**诚实判定**(有规则稳健为正→进 P2;否则记负结果)。
- [ ] **Step 3: 回归 + 提交** 跑全 intraday 测试套件确认无回归;`echo "production/intraday/cache/" >> .gitignore`;提交代码+结果 spec(**不提交 cache/大 parquet**)。
- [ ] **Step 4: 决策门** P1 为正 → 用 brainstorming 起 P2(实时 akshare + Picks 组件);P1 为负 → 记录、止于 P1。

---

## Self-Review
**Spec 覆盖:** 数据获取→T2+T0;进场规则(vwap/低点带/gap/首30)→T1;微结构涨停不可成交→T1(`is_buy_fillable`)+模拟器 skip;选股不变+进/出场口径→T3;复权问题→乘数法(T1 输出 mult,T3 乘到复权 open);验证(净收益 delta+逐年+不可成交占比+open 回归锚)→T4/T5;TDD→T1/T3 失败先行;P2→T5-Step4 草图。✅
**无 Placeholder:** T1/T3 全代码+全测试;T0/T4/T5 为明确 runbook 命令。✅
**类型一致:** `entry_multiplier(day_bars,prev_close,instrument,rule,*,k,g,first_n)->float|None`、`bs_code`、`is_buy_fillable`、`fetch_5min(inst,start,end)`、`prev_close_raw`、`enumerate_trades(scores,top_k,period)->list[dict]`、`simulate(...,rule,...)->dict` 跨任务一致。✅
**风险:** baostock 5min 联网在 T0 阻断验证;乘数法回避复权;`simulate` 的周期聚合是近似(非逐日 NAV),但 open 规则提供回归锚校验量级(T4-Step2)。
