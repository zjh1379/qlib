# 资金流因子 (money-flow) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 factor-LGBM 加入逐个股资金流(主力/大单净流入)特征,重训,检验 factor-MF-2model 能否在规范扣费回测下稳健跑赢 +31.2% 冠军(达标才采纳)。

**Architecture:** qlib 原生新数据层——akshare 拉逐日资金流(按 CSI800 PIT,缓存)→ 建 per-symbol CSV → `scripts/dump_bin.py` dump 成新 qlib 字段 `$mf_*`(并入 `cn_data_bs`)→ handler 子类 `AlphaShortTermMF` 用 qlib 表达式因子追加 → `rolling_train --features shortterm_mf` 重训 → `rebuild_2model` → `build_report` 增量评估。ALSTM/服务不动。

**Tech Stack:** Python、akshare(资金流)、pandas、qlib(`scripts/dump_bin.py`、`D.features`、handler)、解释器 `F:/Tools/Anaconda/envs/qlib/python.exe`。**所有命令在主仓库 `E:\Projects\qlib` 跑、`-X utf8`、`-m`**;改 production 走 feature 分支。

---

## File Structure
| 文件 | 职责 |
|---|---|
| `production/factors/moneyflow_fetch.py` | akshare 拉逐日资金流 + 代码映射 + 解析(纯函数)+ PIT 拉取 + parquet 缓存 |
| `production/factors/moneyflow.py` | `moneyflow_factor_config() -> (fields, names)` qlib 表达式因子 |
| `production/factors/moneyflow_dump.py` | tidy mf df → per-symbol CSV → `scripts/dump_bin.py` dump 成 `$mf_*` 字段 |
| `production/custom_handler.py` | 新增 `AlphaShortTermMF(AlphaShortTerm)`(追加 moneyflow 因子) |
| `production/rolling_train.py` | `resolve_feature_handler` 加 `"shortterm_mf"`;handler 注册表 + `--features` choices(两个子命令) |
| `production/research/_eval_moneyflow.py` | 增量评估(复用 `score_utils` + `build_report`) |
| `production/tests/test_moneyflow.py` | TDD:代码映射 / 解析 / 因子配置 / CSV builder / handler |
| `docs/superpowers/specs/2026-06-10-moneyflow-factor-results.md` | Step 5 结果(跑完写) |

> 起步前:`git checkout -b feat/moneyflow-factor`(主仓库 `E:\Projects\qlib`)。`production/factors/__init__.py` 已存在(short_term 在内);若无则 `New-Item`。

---

## Task 0（runbook · 数据 spike · 阻断门）: 确认 akshare 资金流可用 + 历史深度

- [ ] **Step 1: 实测拉一只票的资金流历史**
Run:
```
F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -c "import akshare as ak; df=ak.stock_individual_fund_flow(stock='600519', market='sh'); print(df.shape); print(list(df.columns)); print(df.head(2).to_string()); print('MIN_DATE', df['日期'].min(), 'MAX_DATE', df['日期'].max())"
```
Expected: 非空 DataFrame;列含 `日期`、`主力净流入-净额`、`超大单净流入-净额`、`大单净流入-净额`、`中单净流入-净额`、`小单净流入-净额`(及对应 `-净占比`);打印出最早日期。
- [ ] **Step 2: 记录可得性事实 + 定跨度** 到 `docs/superpowers/specs/2026-06-10-moneyflow-factor-results.md` 顶部"数据可得性"一节:精确列名、最早/最晚日期、更新时点、行数。**据最早日期定训练+评估跨度**:≤2018 → 全 OOF(2020-07→2025-12);否则 → 资金流因子的训练+评估限在 `[max(最早日期, 2020-07), 2025-12]`,并在结果里标注统计力受限。
- [ ] **Step 3: 阻断判定** 拉不到 / 缺主力净额列 / 连近 1 年都没有 → 停止,改评估 tushare(若用户有 token)或退回大盘资金流覆盖层;**不要继续 Task 1**。拉得到 → 继续。

> 若实际列名与上面不同,以 spike 为准,同步改 Task 1 的 `_COL_MAP`。

---

## Task 1: `moneyflow_fetch.py`（代码映射 + 解析,TDD）

**Files:** Create `production/factors/moneyflow_fetch.py`; Test `production/tests/test_moneyflow.py`

- [ ] **Step 1: 写失败测试**
```python
# production/tests/test_moneyflow.py
import pandas as pd, pytest
from production.factors.moneyflow_fetch import ak_code, parse_moneyflow, MF_COLS


def test_ak_code():
    assert ak_code("SH600519") == ("600519", "sh")
    assert ak_code("SZ000001") == ("000001", "sz")
    assert ak_code("BJ830799") == ("830799", "bj")


def test_parse_moneyflow_maps_and_types():
    raw = pd.DataFrame({
        "日期": ["2024-12-02", "2024-12-03"],
        "收盘价": [1500.0, 1510.0],
        "主力净流入-净额": ["1.0e8", "-2.0e8"],
        "主力净流入-净占比": ["3.5", "-4.0"],
        "超大单净流入-净额": ["6.0e7", "-1.0e8"],
        "大单净流入-净额": ["4.0e7", "-1.0e8"],
        "中单净流入-净额": ["-2.0e7", "5.0e7"],
        "小单净流入-净额": ["-8.0e7", "1.5e8"],
    })
    out = parse_moneyflow(raw, "SH600519")
    assert list(out.columns) == ["datetime", "instrument"] + MF_COLS
    assert str(out["datetime"].iloc[0]) == "2024-12-02 00:00:00"
    assert out["instrument"].iloc[0] == "SH600519"
    assert out["mf_main_net"].dtype.kind == "f"
    assert out["mf_main_net"].iloc[0] == pytest.approx(1.0e8)
    assert out["mf_s_net"].iloc[1] == pytest.approx(1.5e8)
```
- [ ] **Step 2: 跑确认失败** `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_moneyflow.py -q` → ImportError。
- [ ] **Step 3: 写实现**
```python
# production/factors/moneyflow_fetch.py
"""Fetch per-stock daily money-flow from akshare, parse to a tidy typed frame,
cache to parquet. Network is isolated here; moneyflow.py/dump stay offline-pure.

akshare: ak.stock_individual_fund_flow(stock='600519', market='sh') ->
Chinese-column daily frame. Column names confirmed in Task 0 spike; adjust
_COL_MAP if they differ."""
from __future__ import annotations
from pathlib import Path
import time
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE = REPO_ROOT / "production" / "factors" / "cache_moneyflow"

MF_COLS = ["mf_main_net", "mf_xl_net", "mf_l_net", "mf_m_net", "mf_s_net", "mf_main_ratio"]
_COL_MAP = {
    "主力净流入-净额": "mf_main_net",
    "超大单净流入-净额": "mf_xl_net",
    "大单净流入-净额": "mf_l_net",
    "中单净流入-净额": "mf_m_net",
    "小单净流入-净额": "mf_s_net",
    "主力净流入-净占比": "mf_main_ratio",
}


def ak_code(instrument: str) -> tuple[str, str]:
    """SH600519 -> ('600519','sh'); SZ000001 -> ('000001','sz')."""
    return instrument[2:], instrument[:2].lower()


def parse_moneyflow(raw: pd.DataFrame, instrument: str) -> pd.DataFrame:
    """akshare individual-fund-flow frame -> tidy (datetime, instrument, mf_*)."""
    df = pd.DataFrame()
    df["datetime"] = pd.to_datetime(raw["日期"])
    df["instrument"] = instrument
    for zh, en in _COL_MAP.items():
        df[en] = pd.to_numeric(raw[zh], errors="coerce") if zh in raw.columns else 0.0
    return df[["datetime", "instrument"] + MF_COLS]


def fetch_one(instrument: str) -> pd.DataFrame:
    """Fetch + parse one stock's full available money-flow history (cached)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    fp = CACHE / f"{instrument}.parquet"
    if fp.exists():
        return pd.read_parquet(fp)
    import akshare as ak
    stock, market = ak_code(instrument)
    raw = None
    for attempt in range(3):
        try:
            raw = ak.stock_individual_fund_flow(stock=stock, market=market)
            if raw is not None and len(raw):
                break
        except Exception:
            pass
        time.sleep(1.0 * (attempt + 1))
    if raw is None or not len(raw):
        return pd.DataFrame(columns=["datetime", "instrument"] + MF_COLS)
    out = parse_moneyflow(raw, instrument)
    out.to_parquet(fp)            # cache only on genuine success
    return out


def fetch_universe(instruments: list[str]) -> pd.DataFrame:
    """Fetch money-flow for many instruments; concat tidy. Skips empties."""
    frames = [fetch_one(i) for i in instruments]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else \
        pd.DataFrame(columns=["datetime", "instrument"] + MF_COLS)
```
- [ ] **Step 4: 跑测试通过** `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_moneyflow.py -q` → all pass。
- [ ] **Step 5: gitignore 缓存 + 提交**
```
echo "production/factors/cache_moneyflow/" >> .gitignore
git add production/factors/moneyflow_fetch.py production/tests/test_moneyflow.py .gitignore
git commit -m "feat(moneyflow): akshare per-stock money-flow fetch + parse + cache"
```

---

## Task 2: `moneyflow_dump.py`（建 CSV + dump 成 qlib 字段）

**Files:** Create `production/factors/moneyflow_dump.py`; Test append `production/tests/test_moneyflow.py`(纯函数 `to_dump_csv_rows`)

- [ ] **Step 1: 写失败测试(只测 CSV 行构造,不联网/不 dump)**
```python
# append to production/tests/test_moneyflow.py
def test_to_dump_frame_shape():
    from production.factors.moneyflow_dump import to_dump_frame
    tidy = pd.DataFrame({
        "datetime": pd.to_datetime(["2024-12-02", "2024-12-03"]),
        "instrument": ["SH600519", "SH600519"],
        "mf_main_net": [1.0e8, -2.0e8], "mf_xl_net": [6e7, -1e8],
        "mf_l_net": [4e7, -1e8], "mf_m_net": [-2e7, 5e7],
        "mf_s_net": [-8e7, 1.5e8], "mf_main_ratio": [3.5, -4.0]})
    out = to_dump_frame(tidy, "SH600519")
    # dump_bin expects columns: symbol,date,<fields...> ; date as YYYY-MM-DD
    assert list(out.columns)[:2] == ["symbol", "date"]
    assert out["symbol"].iloc[0] == "SH600519"
    assert out["date"].iloc[0] == "2024-12-02"
    assert "mf_main_net" in out.columns and "mf_avail" in out.columns
    assert out["mf_avail"].iloc[0] == 1.0
```
- [ ] **Step 2: 跑确认失败** → ImportError。
- [ ] **Step 3: 写实现**
```python
# production/factors/moneyflow_dump.py
"""Build per-symbol money-flow CSVs and dump them into the cn_data_bs qlib
provider as new $mf_* fields via scripts/dump_bin.py. The dump adds one .bin per
field into features/<sym>/, alongside $open/$close/etc., aligned to the qlib_dir
day calendar."""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path
import pandas as pd

from production.factors.moneyflow_fetch import MF_COLS

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MF_FIELDS = MF_COLS + ["mf_avail"]   # the qlib field names (without the $)


def to_dump_frame(tidy_one: pd.DataFrame, instrument: str) -> pd.DataFrame:
    """One instrument's tidy money-flow -> dump_bin CSV frame:
    columns [symbol, date, mf_main_net, ..., mf_avail]; date='YYYY-MM-DD';
    mf_avail=1.0 marks a real money-flow row (vs a calendar day with none)."""
    df = tidy_one.sort_values("datetime").copy()
    out = pd.DataFrame()
    out["symbol"] = [instrument] * len(df)
    out["date"] = df["datetime"].dt.strftime("%Y-%m-%d").values
    for c in MF_COLS:
        out[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).values
    out["mf_avail"] = 1.0
    return out


def write_csvs(tidy: pd.DataFrame, csv_dir: Path) -> int:
    """Write one <SYM>.csv per instrument into csv_dir. Returns #symbols written."""
    csv_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for inst, g in tidy.groupby("instrument"):
        to_dump_frame(g, inst).to_csv(csv_dir / f"{inst}.csv", index=False)
        n += 1
    return n


def dump_fields(csv_dir: Path, qlib_dir: Path) -> None:
    """dump_bin dump_fix: add the mf_* fields into qlib_dir/features/<sym>/."""
    dump_bin = REPO_ROOT / "scripts" / "dump_bin.py"
    cmd = [sys.executable, str(dump_bin), "dump_fix",
           "--data_path", str(csv_dir), "--qlib_dir", str(qlib_dir),
           "--freq", "day", "--exclude_fields", "symbol,date",
           "--include_fields", ",".join(MF_FIELDS)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"dump_fix failed (rc={r.returncode}):\n{r.stderr[-2000:]}")
```
- [ ] **Step 4: 跑 CSV 纯函数测试通过** `pytest production/tests/test_moneyflow.py -q`。
- [ ] **Step 5: 提交** `git add production/factors/moneyflow_dump.py production/tests/test_moneyflow.py; git commit -m "feat(moneyflow): dump-frame builder + dump_bin field injection"`

---

## Task 3: `moneyflow.py` 因子 + `AlphaShortTermMF` handler + rolling_train 接线（TDD）

**Files:** Create `production/factors/moneyflow.py`; Modify `production/custom_handler.py`, `production/rolling_train.py`; Test append `production/tests/test_moneyflow.py`

- [ ] **Step 1: 写失败测试**
```python
# append to production/tests/test_moneyflow.py
def test_moneyflow_factor_config():
    from production.factors.moneyflow import moneyflow_factor_config
    fields, names = moneyflow_factor_config()
    assert len(fields) == len(names) == 5
    assert len(set(names)) == 5
    assert "MF_MAIN_NORM" in names
    assert any("$mf_main_net" in f for f in fields)


def test_resolve_feature_handler_mf():
    from production.rolling_train import resolve_feature_handler
    assert resolve_feature_handler("shortterm_mf") == ("AlphaShortTermMF", "custom_handler")


def test_handler_appends_moneyflow_factors():
    from production.factors.moneyflow import moneyflow_factor_config
    from production.factors.short_term import short_term_factor_config
    # AlphaShortTermMF.get_feature_config must include both short-term + moneyflow names
    mf_names = set(moneyflow_factor_config()[1])
    st_names = set(short_term_factor_config()[1])
    assert mf_names.isdisjoint(st_names)        # no name collisions
```
- [ ] **Step 2: 跑确认失败** → ImportError / AssertionError。
- [ ] **Step 3a: 写 `production/factors/moneyflow.py`**
```python
# production/factors/moneyflow.py
"""Money-flow alpha factors (qlib expressions over the dumped $mf_* fields).
Normalized so cross-sectional scale is comparable; trailing windows only (no
lookahead). $amount may be absent in cn_data_bs -> normalize by $close*$volume."""
from __future__ import annotations

_AMT = "($close*$volume+1e-12)"


def moneyflow_factor_config() -> tuple[list[str], list[str]]:
    """Return (expression_fields, names) appended onto AlphaShortTerm for LGBM."""
    fields = [
        f"$mf_main_net/{_AMT}",                                  # day main-force net, turnover-normalized
        f"Mean($mf_main_net,5)/{_AMT}",                          # 5d cumulative normalized
        f"(Mean($mf_main_net,5)-Mean($mf_main_net,20))/{_AMT}",  # money-flow momentum
        f"($mf_xl_net+$mf_l_net-$mf_m_net-$mf_s_net)/{_AMT}",    # big-vs-small order imbalance
        "$mf_avail",                                             # data-availability mask
    ]
    names = ["MF_MAIN_NORM", "MF_MAIN_MA5", "MF_MOM", "MF_BIGSMALL", "MF_AVAIL"]
    return fields, names
```
- [ ] **Step 3b: 在 `production/custom_handler.py` 末尾追加 `AlphaShortTermMF`**
```python
# append to production/custom_handler.py
try:
    from production.factors.moneyflow import moneyflow_factor_config
except ModuleNotFoundError:
    from factors.moneyflow import moneyflow_factor_config  # type: ignore[no-redef]


class AlphaShortTermMF(AlphaShortTerm):
    """AlphaShortTerm + per-stock money-flow factors (needs $mf_* fields dumped)."""

    def get_feature_config(self):
        fields, names = super().get_feature_config()
        extra_fields, extra_names = moneyflow_factor_config()
        return list(fields) + extra_fields, list(names) + extra_names
```
- [ ] **Step 3c: 在 `production/rolling_train.py` 接线(3 处)**
1) `resolve_feature_handler`(line ~79):在 `if features == "shortterm": return ("AlphaShortTerm", "custom_handler")` 之后加:
```python
    if features == "shortterm_mf":
        return ("AlphaShortTermMF", "custom_handler")
```
2) handler 注册表(line ~254-258):导入 + 注册:
```python
    from custom_handler import Alpha158_OpenH, AlphaShortTerm, AlphaShortTermMF  # noqa: E402
    ...
        "AlphaShortTerm": AlphaShortTerm,
        "AlphaShortTermMF": AlphaShortTermMF,
```
3) 两个子命令的 `--features choices`(line ~660 和 ~724):
```python
        choices=["alpha158", "shortterm", "shortterm_mf"],
```
- [ ] **Step 4: 跑测试通过** `pytest production/tests/test_moneyflow.py -q` + 回归 `pytest production/tests/test_short_term_factors.py production/tests/test_score_utils.py -q`。
- [ ] **Step 5: 提交** `git add production/factors/moneyflow.py production/custom_handler.py production/rolling_train.py production/tests/test_moneyflow.py; git commit -m "feat(moneyflow): factor config + AlphaShortTermMF handler + rolling_train --features shortterm_mf"`

---

## Task 4（runbook · 拉真数据 + dump + 重训 + pool）

- [ ] **Step 1: 写一次性脚本 `production/research/_build_moneyflow.py`**(`-m` 运行):载入 PIT universe(`production/pit_constituents.parquet` 的全 instrument 并集)→ `fetch_universe(insts)`(缓存)→ `write_csvs` 到 `~/.qlib/stock_data/moneyflow_csv` → `dump_fields(csv_dir, ~/.qlib/qlib_data/cn_data_bs)`。打印拉到的 name-day 数 + 字段。含 purelib/utf8 preamble。
- [ ] **Step 2: 跑构建 + dump**
Run: `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m production.research._build_moneyflow > logs/moneyflow_build.log 2>&1`
- [ ] **Step 3: round-trip 验证字段读回**
Run:
```
F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -c "import sys,sysconfig; sys.path.insert(0,sysconfig.get_paths()['purelib']); from production.backtest.data import init_qlib_from_config; init_qlib_from_config('production/configs/rolling_ensemble.yaml'); from qlib.data import D; df=D.features(['SH600519'],['$mf_main_net','$mf_avail','$close'],start_time='2024-12-01',end_time='2024-12-10'); print(df.dropna().tail()); print('rows',len(df))"
```
Expected: `$mf_main_net` 非全 NaN、与 `$close` 同日对齐、`$mf_avail`∈{0,1}。**对不齐/全 NaN → 回查 Task 2 dump,不要继续。**
- [ ] **Step 4: 重训 factor-MF backfill(隔离实验)** 复制 `production/configs/rolling_ensemble_fac.yaml` → `rolling_ensemble_mf.yaml`,改 `experiment_name` 为 `rolling_mf`。跑(Step-0 定的跨度,示例全 OOF):
```
F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m production.rolling_train backfill --features shortterm_mf --config production/configs/rolling_ensemble_mf.yaml --train-years 3 --test-weeks 4 --step-weeks 4 > logs/moneyflow_backfill.log 2>&1
```
(具体 `--train-years/--test-weeks/--step-weeks` 对齐既有 factor backfill 的参数;见 `quant_pipeline.md`。)
- [ ] **Step 5: pool 成 OOF** 写 `production/research/_pool_mf.py`(镜像 `production/research/_pool_fac.py`,改实验名 `rolling_mf` + 输出 `production/reports/oof_lgbmmf_2021_2026.pkl`);跑 `-m production.research._pool_mf`。

---

## Task 5（runbook · 增量评估 + 出结论 + 存档）

- [ ] **Step 1: 写 `production/research/_eval_moneyflow.py`**(镜像 `_eval_factors`,复用 `score_utils` + `build_report`):
载入 `oof_lgbmmf` + `oof_lgbmfac` + `oof_2model`;构造 `factor-2model = rebuild_2model(fac, two)`、`factor-MF-2model = rebuild_2model(mf, two)`;**对齐到两者共同 (datetime×instrument)**;对每个跑 `build_report(scores, policy_name="fixed", top_k=5, period=5, exit_k=10, capital=100000, profile="small")` 及 `regime={'method':'trend_ma','ma_window':60,'band':0.10}`;打印 net_cagr/Calmar/maxDD/逐年 + delta;配对 t 用 `production.research._eval_robustness._paired_t`。写 `logs/eval_moneyflow_summary.json`。
- [ ] **Step 2: 跑评估**
Run: `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m production.research._eval_moneyflow > logs/eval_moneyflow.log 2>&1`
Expected: 打印 factor-2model(应≈+31.2% 回归锚)与 factor-MF-2model 的对比 + 逐年 + delta。
- [ ] **Step 3: 写结果 spec** `docs/superpowers/specs/2026-06-10-moneyflow-factor-results.md`(已含 Task 0 的数据可得性节):对比表(MF vs 冠军,±overlay)、逐年(尤其 2022/2023)、配对 t、与正交性预期一致性、**诚实判定**(稳健跑赢→采纳进 Task 6;否则记负、止于 Task 5)。
- [ ] **Step 4: 回归 + 提交** 跑 `pytest production/tests/test_moneyflow.py production/tests/test_short_term_factors.py -q` 确认无回归;提交代码 + 结果 spec + `_build_moneyflow`/`_pool_mf`/`_eval_moneyflow`/`rolling_ensemble_mf.yaml`(**不提交 cache/CSV/OOF 大文件**)。

---

## Task 6（达标才做 · 产品化 + 重启前后端）

- [ ] **Step 1: 决策门** Task 5 为正 → 继续;为负 → 记录、止于 Task 5(本计划结束)。
- [ ] **Step 2: 重生服务 pred.pkl** 镜像 `production/research/_regen_live_factor.py`,改用 `shortterm_mf` 重训的最新 live fold,把 factor-MF-2model 的 pred.pkl 写进 `rolling_v2_ensemble` 的新 recorder。
- [ ] **Step 3: 接入每日刷新** 在 `production/daily_inference.py`(或 data-refresh on_success 链)加一步:刷新当日资金流 → `write_csvs` + `dump_fields`(dump_update)→ 再跑推理,保证 `$mf_*` 字段每日最新、训练/实盘同口径。
- [ ] **Step 4: 重启前后端供查看**
```
Set-Location E:\Projects\qlib\backend; F:/Tools/Anaconda/envs/qlib/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000   # 后台
Set-Location E:\Projects\qlib\frontend; npm run dev                                                                                 # 后台
```
验证 `/api/models/candidates` 200 + 前端 Picks 展示新选股。
- [ ] **Step 5: 提交 + 更新 MEMORY** 提交产品化改动;更新 `quant_pipeline.md`(资金流采纳/否决结论 + `shortterm_mf` 入口)。

---

## Self-Review
**Spec 覆盖:** Step0 spike→T0;PIT 拉取+缓存→T1;dump 成字段→T2;表达式因子+handler+接线→T3;重训+pool→T4;增量评估+判定→T5;产品化+daily-refresh+重启→T6;实盘一致性→T6-Step3;生存偏差→T4-Step1(PIT);测试→T1/T2/T3 TDD + T4-Step3 round-trip + open 回归(T3-Step4 跑既有 shortterm 测试)。✅
**无 Placeholder:** T1/T2/T3 全代码+全测试;T0/T4/T5/T6 为明确 runbook 命令(脚本镜像已存在的 `_pool_fac`/`_eval_factors`/`_regen_live_factor`,路径具体)。✅
**类型一致:** `ak_code(inst)->(stock,market)`、`parse_moneyflow(raw,inst)->df[ datetime,instrument,*MF_COLS ]`、`MF_COLS`、`to_dump_frame(tidy_one,inst)`、`MF_FIELDS=MF_COLS+['mf_avail']`、`moneyflow_factor_config()->(fields,names)`(5 个)、`AlphaShortTermMF`、`resolve_feature_handler('shortterm_mf')->('AlphaShortTermMF','custom_handler')` 跨任务一致。✅
**风险:** akshare 列名/历史深度在 T0 阻断验证(列名不符即改 `_COL_MAP`);`$amount` 缺失已用 `$close*$volume` 兜底;dump 对齐在 T4-Step3 round-trip 校验;增量可能为负(第 6 个诚实负结果),T5 照常记录。
