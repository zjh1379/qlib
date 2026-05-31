# P2a 可信长窗口回填 + 多 regime 评估 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (for the CODE tasks T1–T4) or superpowers:executing-plans. The EXECUTION tasks (T5–T6) are long background backfills — run them via the runbook, not a synchronous subagent. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 用粗粒度（半年/年度）走查回填产出 2021–2026 的连续 OOF 预测，再用已建好的 P0/P1 净收益引擎做多 regime 评估，第一次得到现有模型**统计可信**的真实战绩。

**Architecture:** 复用现有 `rolling_train`/`run_split`/`walk_forward`/pooling 机器。新增三处小改：(1) 回填支持 `--step-weeks` 粗步长 + `--test-weeks` 长测试窗（一次训练预测数月）；(2) 回填**可断点续跑**（已存在 recorder 的折跳过）；(3) `pool_range` 把一段日期内所有折的逐模型 OOF 拼成一条连续序列。评估直接用 `production/backtest/run.py` + `sweep.py`（无新代码）。

**Tech Stack:** Python, pandas, qlib, pytest。解释器 `F:/Tools/Anaconda/envs/qlib/python.exe`。内存安全：现有 per-process RSS(10GB)/commit(85%) 看门狗 + `production/safety_watchdog.py`。

---

## 关键设计决策（已与用户确认）
- **顺序**：LGBM 基线先行（轻、~2h、立刻可信）；再 ALSTM+TRA **年度折**增量（~5 折，串行、断点续、watchdog 下、后台 ~0.5–1 天）。
- **数据起点约束**：qlib 数据自 2018-01；为最大化 1–5d 的可评估区间，回填**各 horizon 统一用 3 年训练窗**（`--train-years-override 3`），评估窗约 **2021–2026**。
- **连续 OOF**：取 `step_weeks == test_weeks`（相邻折测试窗首尾相接、不重叠），cadence：LGBM 26 周、神经 52 周。
- **不训练新因子**：P2a 只回填现有 Alpha158/360 模型；短线因子是 P2b（LGBM-only、便宜），神经的昂贵回填只在 P2a 做一次。

## 文件结构
| 文件 | 改动 |
|---|---|
| `production/rolling_train.py` | run_once 加 `test_weeks_override`/`train_years_override`/`skip_if_exists`；run_backfill 加 `step_weeks`/override 透传；CLI flags；新纯函数 `backfill_fold_end_dates` |
| `production/backfill_pool.py`（新） | `assemble_score`(纯) + `pool_range(start,end,...)` 拼接连续 OOF → 一个 pred pickle |
| `production/backfill_longwindow.py`（新） | 编排脚本：预检 + LGBM(26w) → 神经(52w) 串行、resumable、日志落文件 |
| `production/tests/test_backfill_schedule.py`（新） | `backfill_fold_end_dates` + skip 谓词 |
| `production/tests/test_backfill_pool.py`（新） | `assemble_score` 纯函数 |
| `docs/superpowers/specs/2026-05-31-net-return-results.md` | 追加长窗口多 regime 结果 |

---

## Phase: 代码（T1–T4，TDD，可子代理执行）

### Task 1: 粗步长 + 长测试窗 + 训练窗覆盖（回填参数化）
**Files:** Modify `production/rolling_train.py`; Test `production/tests/test_backfill_schedule.py`

- [ ] **Step 1: 写失败测试**
```python
# production/tests/test_backfill_schedule.py
from datetime import date
from production.rolling_train import backfill_fold_end_dates


def test_weekly_step_default():
    # 2021-01-01 is a Friday-anchored walk; step_weeks=1 -> weekly
    out = backfill_fold_end_dates(date(2021, 1, 1), date(2021, 2, 1), step_weeks=1)
    assert out[0].weekday() == 4          # all Fridays
    assert (out[1] - out[0]).days == 7
    assert all(d.weekday() == 4 for d in out)


def test_semiannual_step_contiguous():
    out = backfill_fold_end_dates(date(2021, 1, 1), date(2026, 1, 1), step_weeks=26)
    assert (out[1] - out[0]).days == 26 * 7
    assert 8 <= len(out) <= 12            # ~10 folds over 5y


def test_annual_step():
    out = backfill_fold_end_dates(date(2021, 1, 1), date(2026, 1, 1), step_weeks=52)
    assert (out[1] - out[0]).days == 52 * 7
    assert 4 <= len(out) <= 6             # ~5 folds
```

- [ ] **Step 2: 运行确认失败**
Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_backfill_schedule.py::test_semiannual_step_contiguous -v`
Expected: FAIL (`ImportError: cannot import name 'backfill_fold_end_dates'`)

- [ ] **Step 3: 实现**
在 `production/rolling_train.py` 顶部附近加纯函数，并参数化 `run_backfill`/`run_once`：
```python
from datetime import date, timedelta

def backfill_fold_end_dates(start: date, end: date, step_weeks: int = 1) -> list[date]:
    """Friday-anchored fold end-dates from start..end, stepping step_weeks each time."""
    days_to_friday = (4 - start.weekday()) % 7
    cursor = start + timedelta(days=days_to_friday)
    out: list[date] = []
    while cursor <= end:
        out.append(cursor)
        cursor += timedelta(days=step_weeks * 7)
    return out
```
修改 `run_once(cfg, end_date, ...)` 增加可选参数（在构建每个 HorizonConfig 前覆盖）：
```python
def run_once(cfg, end_date, *, test_weeks_override=None, train_years_override=None,
             skip_if_exists=False, only_models=None):
    if test_weeks_override or train_years_override:
        new_h = []
        for h in cfg.horizons:
            new_h.append(replace(h,
                test_weeks=test_weeks_override or h.test_weeks,
                train_years=train_years_override or h.train_years))
        cfg = replace(cfg, horizons=new_h)   # dataclasses.replace; import it
    # ... existing body ...
```
（`HorizonConfig`/`RollingConfig` 若非 dataclass，则改为手动复制字段；用 `from dataclasses import replace`。）
修改 `run_backfill` 用纯函数生成折日期 + 透传 override：
```python
def run_backfill(cfg, start, end, *, step_weeks=1, test_weeks_override=None,
                 train_years_override=None, skip_if_exists=True):
    paths, failures = [], []
    for cursor in backfill_fold_end_dates(start, end, step_weeks):
        try:
            paths.append(run_once(cfg, cursor,
                test_weeks_override=test_weeks_override,
                train_years_override=train_years_override,
                skip_if_exists=skip_if_exists))
        except Exception as exc:
            failures.append((cursor, str(exc)))
    if failures:
        log.warning("backfill folds failed: %s", failures)
    return paths
```
在 backfill 子命令 argparse 加：`--step-weeks`(int,默认1)、`--test-weeks`(int,默认None)、`--train-years`(int,默认None)，并传入 `run_backfill`。

- [ ] **Step 4: 运行确认通过**
Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_backfill_schedule.py -v`
Expected: 3 passed (skip 谓词测试在 Task 2 追加)

- [ ] **Step 5: 提交**
```bash
git add production/rolling_train.py production/tests/test_backfill_schedule.py
git commit -m "feat(backfill): coarse step-weeks + long test-window + train-years override"
```

---

### Task 2: 可断点续跑（跳过已存在的折）
**Files:** Modify `production/rolling_train.py`; Test `production/tests/test_backfill_schedule.py`

- [ ] **Step 1: 追加失败测试**
```python
# append to production/tests/test_backfill_schedule.py
from datetime import date
from production.rolling_train import fold_recorders_complete


class _FakeExp:
    def __init__(self, names): self._names = names
    def list_recorders(self):
        return [type("R", (), {"info": {"name": n}})() for n in self._names]


def test_fold_complete_true_when_all_present():
    names = [f"{m}_{h}_2026-01-02" for m in ("lgbm","alstm","tra") for h in ("1d","5d","20d")]
    assert fold_recorders_complete(_FakeExp(names), date(2026,1,2),
                                   ("lgbm","alstm","tra"), ("1d","5d","20d")) is True


def test_fold_complete_false_when_missing_one():
    names = [f"lgbm_{h}_2026-01-02" for h in ("1d","5d","20d")]  # only lgbm
    assert fold_recorders_complete(_FakeExp(names), date(2026,1,2),
                                   ("lgbm","alstm","tra"), ("1d","5d","20d")) is False
```

- [ ] **Step 2: 运行确认失败**
Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_backfill_schedule.py -k fold_complete -v`
Expected: FAIL (`ImportError: ... 'fold_recorders_complete'`)

- [ ] **Step 3: 实现**
```python
# in production/rolling_train.py
def _rec_name(r) -> str:
    info = getattr(r, "info", {})
    if isinstance(info, dict) and info.get("name"):
        return info["name"]
    return getattr(r, "name", "") or ""

def fold_recorders_complete(exp, end_date, models, horizons) -> bool:
    """True if every <model>_<horizon>_<end_date> recorder already exists."""
    have = {_rec_name(r) for r in exp.list_recorders()}
    es = end_date.isoformat()
    needed = {f"{m}_{h}_{es}" for m in models for h in horizons}
    return needed.issubset(have)
```
在 `run_once` 开头（当 `skip_if_exists`）查：用 `R.get_exp(experiment_name=cfg.experiment_name)`，调 `fold_recorders_complete(exp, end_date, only_models or DEFAULT_MODELS, [h.name for h in cfg.horizons])`；若 True，`log.info("skip existing fold %s", end_date)` 并直接返回该折已有的 `examples/mlruns/pred_<date>.pkl`（若不存在则继续训练）。

- [ ] **Step 4: 运行确认通过**
Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_backfill_schedule.py -v`
Expected: all passed

- [ ] **Step 5: 提交**
```bash
git add production/rolling_train.py production/tests/test_backfill_schedule.py
git commit -m "feat(backfill): resumable — skip folds whose recorders already exist"
```

---

### Task 3: pool_range — 连续 OOF 拼接
**Files:** Create `production/backfill_pool.py`; Test `production/tests/test_backfill_pool.py`

- [ ] **Step 1: 写失败测试**
```python
# production/tests/test_backfill_pool.py
import pandas as pd
from production.backfill_pool import assemble_score


def _series(d):
    idx = pd.MultiIndex.from_tuples(list(d.keys()), names=["datetime", "instrument"])
    return pd.Series(list(d.values()), index=idx)


def test_assemble_score_ranks_1d_5d_only_and_dedups():
    t = pd.Timestamp("2026-01-02")
    base = pd.DataFrame({
        "lgbm_1d": _series({(t, "A"): 0.9, (t, "B"): 0.1}),
        "lgbm_5d": _series({(t, "A"): 0.8, (t, "B"): 0.2}),
        "lgbm_20d": _series({(t, "A"): -9.0, (t, "B"): 9.0}),  # must be ignored by score
    })
    out = assemble_score(base, ewma_alpha=1.0)  # alpha=1 -> no smoothing
    a = out.xs(t, level="datetime").loc["A", "score"]
    b = out.xs(t, level="datetime").loc["B", "score"]
    assert a > b                      # A better on 1d+5d -> higher score
    assert "score" in out.columns


def test_assemble_score_dedup_keep_last():
    t = pd.Timestamp("2026-01-02")
    base = pd.DataFrame({"lgbm_1d": _series({(t, "A"): 1.0})})
    dup = pd.concat([base, base])     # duplicated index row
    out = assemble_score(dup, ewma_alpha=1.0)
    assert len(out.xs(t, level="datetime")) == 1
```

- [ ] **Step 2: 运行确认失败**
Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_backfill_pool.py -v`
Expected: FAIL (`No module named 'production.backfill_pool'`)

- [ ] **Step 3: 实现**
```python
# production/backfill_pool.py
"""Pool per-fold OOF predictions across a date range into one continuous
(datetime,instrument)->score series for long-window backtesting."""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS = ("lgbm", "alstm", "tra")
HORIZONS = ("1d", "5d", "20d")


def assemble_score(base: pd.DataFrame, ewma_alpha: float = 0.5) -> pd.DataFrame:
    """De-dup (keep last), score = -mean(rank over 1d+5d cols), per datetime.
    EWMA-smooth score per instrument (alpha=1.0 disables smoothing)."""
    base = base[~base.index.duplicated(keep="last")].sort_index()
    base.index = base.index.set_names(["datetime", "instrument"])
    score_cols = [c for c in base.columns if not c.endswith("_20d")]
    ranks = base[score_cols].groupby(level="datetime").rank(ascending=False, method="min")
    base["score"] = -ranks.mean(axis=1, skipna=True)
    if ewma_alpha < 1.0:
        base["score"] = (base["score"].groupby(level="instrument")
                         .transform(lambda s: s.ewm(alpha=ewma_alpha).mean()))
    return base


def _rec_name(r) -> str:
    info = getattr(r, "info", {})
    if isinstance(info, dict) and info.get("name"):
        return info["name"]
    return getattr(r, "name", "") or ""


def pool_range(start: date, end: date, *, models=DEFAULT_MODELS,
               config_path: str = "production/configs/rolling_ensemble.yaml",
               out_path: str | None = None) -> Path:
    """Load every <model>_<h>_<fold> recorder with fold in [start,end], concat each
    model_horizon column across folds, assemble score, write one long pred pickle."""
    from qlib.workflow import R
    from production.rolling_train import load_config, init_qlib
    cfg = load_config(Path(config_path))
    init_qlib(cfg)
    exp = R.get_exp(experiment_name=cfg.experiment_name)
    recs = exp.list_recorders()
    recs = list(recs.values()) if isinstance(recs, dict) else recs

    pat = re.compile(r"^(%s)_(1d|5d|20d)_(\d{4}-\d{2}-\d{2})$" % "|".join(models))
    cols: dict[str, list[pd.Series]] = {}
    for r in recs:
        m = pat.match(_rec_name(r))
        if not m:
            continue
        fold = date.fromisoformat(m.group(3))
        if not (start <= fold <= end):
            continue
        model, h = m.group(1), m.group(2)
        try:
            s = r.load_object(f"pred_{h}.pkl")
        except Exception:
            continue
        if isinstance(s, pd.DataFrame):
            s = s["score"] if "score" in s.columns else s.iloc[:, 0]
        cols.setdefault(f"{model}_{h}", []).append(s)

    if not cols:
        raise SystemExit("no matching recorders in range")
    merged = {k: pd.concat(v).sort_index() for k, v in cols.items()}
    base = pd.concat([s.rename(k) for k, s in merged.items()], axis=1).sort_index()
    base = assemble_score(base)
    out = Path(out_path or REPO_ROOT / "production" / "reports" /
               f"oof_{start.isoformat()}_{end.isoformat()}.pkl")
    out.parent.mkdir(parents=True, exist_ok=True)
    base.to_pickle(out)
    print(f"wrote {out} rows={len(base)} span={base.index.get_level_values('datetime').min()}"
          f"..{base.index.get_level_values('datetime').max()}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--models", default="lgbm,alstm,tra")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pool_range(date.fromisoformat(a.start), date.fromisoformat(a.end),
               models=tuple(a.models.split(",")), out_path=a.out)
```

- [ ] **Step 4: 运行确认通过**
Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_backfill_pool.py -v`
Expected: 2 passed

- [ ] **Step 5: 提交**
```bash
git add production/backfill_pool.py production/tests/test_backfill_pool.py
git commit -m "feat(backfill): pool_range -> continuous long-window OOF series"
```

---

### Task 4: 编排脚本 + 预检
**Files:** Create `production/backfill_longwindow.py`; Test `production/tests/test_backfill_schedule.py`(追加一个命令组装测试)

- [ ] **Step 1: 追加失败测试**
```python
# append to production/tests/test_backfill_schedule.py
from production.backfill_longwindow import build_backfill_cmd


def test_build_backfill_cmd_lgbm():
    cmd = build_backfill_cmd("lgbm", "2021-01-01", "2026-01-01", step_weeks=26,
                             test_weeks=26, train_years=3)
    s = " ".join(cmd)
    assert "production.rolling_train" in s and "backfill" in s
    assert "--only-models lgbm" in s
    assert "--step-weeks 26" in s and "--test-weeks 26" in s and "--train-years 3" in s
```

- [ ] **Step 2: 运行确认失败**
Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_backfill_schedule.py -k build_backfill_cmd -v`
Expected: FAIL (`No module named 'production.backfill_longwindow'`)

- [ ] **Step 3: 实现**
```python
# production/backfill_longwindow.py
"""Orchestrate the long-window backfill: LGBM (semi-annual) then ALSTM+TRA
(annual), sequentially, resumable, under safety_watchdog. Run in background.

PRE-FLIGHT (do manually before launching — see plan T5):
  1. Ensure pagefile >= 64GB.   2. Start safety_watchdog in another terminal:
     F:/Tools/Anaconda/envs/qlib/python.exe -m production.safety_watchdog
  3. Close heavy browser tabs.

Usage (background):
  F:/Tools/Anaconda/envs/qlib/python.exe -m production.backfill_longwindow \
     --start 2021-01-01 --end 2026-01-01 > logs/backfill_long.log 2>&1
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PY = sys.executable
REPO = Path(__file__).resolve().parent.parent


def build_backfill_cmd(model: str, start: str, end: str, *, step_weeks: int,
                       test_weeks: int, train_years: int) -> list[str]:
    return [PY, "-m", "production.rolling_train", "backfill",
            "--start", start, "--end", end,
            "--only-models", model,
            "--step-weeks", str(step_weeks),
            "--test-weeks", str(test_weeks),
            "--train-years", str(train_years)]


def _run(cmd: list[str]) -> int:
    print(">>>", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2026-01-01")
    ap.add_argument("--skip-neural", action="store_true",
                    help="LGBM baseline only (fast, safe)")
    a = ap.parse_args()

    # 1) LGBM semi-annual (light, fast). Resumable via run_once skip_if_exists.
    rc = _run(build_backfill_cmd("lgbm", a.start, a.end,
                                 step_weeks=26, test_weeks=26, train_years=3))
    if rc != 0:
        print("LGBM backfill returned", rc)
    if a.skip_neural:
        return 0
    # 2) Neural annual (heavy) — one model at a time, watchdog protects memory.
    for model in ("alstm", "tra"):
        rc = _run(build_backfill_cmd(model, a.start, a.end,
                                     step_weeks=52, test_weeks=52, train_years=3))
        if rc != 0:
            print(f"{model} backfill returned", rc, "(continuing; resumable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 运行确认通过**
Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_backfill_schedule.py -v`
Expected: all passed

- [ ] **Step 5: 提交**
```bash
git add production/backfill_longwindow.py production/tests/test_backfill_schedule.py
git commit -m "feat(backfill): long-window orchestrator (LGBM then neural, resumable)"
```

---

## Phase: 执行（T5–T6，后台 runbook，非同步子代理）

### Task 5: 跑 LGBM 长回填 + 长窗口基线评估
- [ ] **预检**：确认 pagefile ≥ 64GB；新开终端起看门狗：`F:/Tools/Anaconda/envs/qlib/python.exe -m production.safety_watchdog`
- [ ] **冒烟（1 折）**：`... -m production.rolling_train run-once --end-date 2021-07-02 --only-models lgbm --test-weeks 26 --train-years 3` —— 确认产出 `lgbm_*_2021-07-02` recorder 且无异常。
- [ ] **LGBM 全量（后台，~2h）**：`F:/Tools/Anaconda/envs/qlib/python.exe -m production.backfill_longwindow --start 2021-01-01 --end 2026-01-01 --skip-neural > logs/backfill_lgbm.log 2>&1`
- [ ] **拼 OOF**：`... -m production.backfill_pool --start 2021-01-01 --end 2026-01-01 --models lgbm --out production/reports/oof_lgbm_2021_2026.pkl`
- [ ] **长窗口评估**（复用 P0/P1）：
  - 诚实日频基线：`... -m production.backtest.run --pred-file production/reports/oof_lgbm_2021_2026.pkl --policy daily --top-k 30 --capital 100000 --profile small --out production/reports/long_lgbm_daily.json`
  - 扫描：`... -m production.backtest.sweep --pred-file production/reports/oof_lgbm_2021_2026.pkl --out production/reports/long_lgbm_sweep.csv`
  - 最优配置 + 各 regime 净 IR（看 JSON `regimes`）。
- [ ] **验收**：`long_lgbm_daily.json` 与最优配置的 `regimes[*].net_ir` 至少多数 > 0；记录 net_cagr/net_ir/turnover/各 regime。

### Task 6: 跑神经年度回填 + 三模型长评估 + 结果文档
- [ ] **神经全量（后台，~0.5–1天，watchdog 下）**：`F:/Tools/Anaconda/envs/qlib/python.exe -m production.backfill_longwindow --start 2021-01-01 --end 2026-01-01 > logs/backfill_neural.log 2>&1`（断点续：重跑会跳过已完成折）
- [ ] **监控**：期间看 `logs/watchdog.log` 与 `logs/backfill_neural.log`；若某折被 watchdog 杀，重跑命令会跳过已完成折继续。
- [ ] **拼三模型 OOF**：`... -m production.backfill_pool --start 2021-01-01 --end 2026-01-01 --models lgbm,alstm,tra --out production/reports/oof_3model_2021_2026.pkl`
- [ ] **长评估**：对 `oof_3model_2021_2026.pkl` 重跑 run(daily 基线) + sweep + 最优配置 + `--neutralize` 变体；各 regime 净 IR。
- [ ] **结果文档**：在 `docs/superpowers/specs/2026-05-31-net-return-results.md` 追加 "长窗口（2021–2026）多 regime" 一节：LGBM-only vs 三模型、诚实日频 vs 最优降换手、各 regime 净 IR、中性化在长窗口的真实效果（对照 P1 短窗口的"有害"结论是否反转）、统计可信度评述。
- [ ] **回归 + 提交**：跑全 backtest 套件（应仍 29 passed）+ 新增 schedule/pool 测试；`git add production/reports/long_*.json production/reports/long_*sweep.csv production/reports/oof_*.pkl docs/superpowers/specs/2026-05-31-net-return-results.md`，提交 `docs(p2a): long-window multi-regime honest evaluation results`。

---

## Self-Review
**Spec 覆盖**：可信长评估(design §3.3) → T1(长测试窗/粗步长)+T3(拼接)+T5/T6(评估)。算力安全(§10) → 复用看门狗 + 串行 + 断点续 + LGBM先行/神经年度(用户确认范围)。多 regime → P0/P1 `net_regime` 直接用。✅
**Placeholder 扫描**：T1–T4 均给完整代码与确切命令；T5/T6 为执行 runbook，命令确切。注意 T1 的 dataclass `replace` 取决于 `HorizonConfig/RollingConfig` 是否 dataclass——实现者须先确认（Step 3 已注明回退方案）。
**类型一致**：`backfill_fold_end_dates(start,end,step_weeks)`、`fold_recorders_complete(exp,end,models,horizons)`、`assemble_score(base,ewma_alpha)`、`pool_range(start,end,models,config_path,out_path)`、`build_backfill_cmd(model,start,end,*,step_weeks,test_weeks,train_years)` 在测试与调用处一致。评估复用 P0/P1 已存在的 `run.py`/`sweep.py`（`--pred-file` 接 OOF pickle）。✅
**风险**：T1 改 `run_once` 签名——须保持对现有周度生产调用（无 override）向后兼容（所有新参数带默认值，已满足）。

## 后续
- **P2b**：amount/turn 数据 + 短线因子 + `AlphaShortTerm` LGBM 回填（便宜）+ 在本计划的长窗口上做因子增量评估。
- **P3/P4**：见 design §11。
