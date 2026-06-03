# P2b 短线因子 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (CODE tasks T1–T3) or executing-plans. T4 is an execution/runbook (~2h LGBM backfill). Steps use checkbox (`- [ ]`).

**Goal:** 给 LGBM 的 Alpha158 之上追加一批非冗余、现有数据可算的短线因子，长窗口检验是否加厚 alpha。

**Architecture:** `short_term.py` 提供纯函数 `short_term_factor_config()`（6 个 qlib 表达式因子）；`custom_handler.AlphaShortTerm` 继承 `Alpha158_OpenH` 把因子追加进 `get_feature_config()`；`rolling_train` 加 `--features {alpha158,shortterm}` 选择 LGBM handler（默认 alpha158，向后兼容）。评估用 P2a 的半年折 LGBM-only 回填 + P0/P1 引擎对比。

**Tech Stack:** Python, qlib (Alpha158DL/表达式 Gt/Sum/Ref/Mean/Std 已确认可用), pytest。解释器 `F:/Tools/Anaconda/envs/qlib/python.exe`；仓库根 `E:\Projects\qlib`；pytest 从根跑；改 production 代码 → 走 feature 分支。

---

## 关键约定
- 因子全部**尾部表达式**（`Ref(...,+k)` = 过去），无前视。
- handler 扩展：`AlphaShortTerm.get_feature_config()` = `super().get_feature_config()` (Alpha158 的 158) + `short_term_factor_config()`。
- 向后兼容：`features` 默认 `"alpha158"`，现有 LGBM 路径行为不变。

## 文件结构
| 文件 | 改动 |
|---|---|
| `production/factors/__init__.py` | 新（空包标记） |
| `production/factors/short_term.py` | 新：`short_term_factor_config()` |
| `production/custom_handler.py` | 改：加 `AlphaShortTerm` |
| `production/rolling_train.py` | 改：`resolve_feature_handler()` + `train_lgbm_horizon`/`run_once`/`run_backfill`/CLI 加 `features` |
| `production/tests/test_short_term_factors.py` | 新 |

---

### Task 1: short_term.py 因子定义

**Files:** Create `production/factors/__init__.py`, `production/factors/short_term.py`; Test `production/tests/test_short_term_factors.py`

- [ ] **Step 1: 写失败测试**
```python
# production/tests/test_short_term_factors.py
from production.factors.short_term import short_term_factor_config


def test_returns_fields_and_names_aligned():
    fields, names = short_term_factor_config()
    assert len(fields) == len(names) == 6
    assert names == ["OVNGAP", "OVNGAP_MA5", "OVNGAP_STD5",
                     "AMT_SURGE", "LIMITUP10_CNT20", "LIMITUP20_CNT20"]


def test_overnight_gap_expr():
    fields, names = short_term_factor_config()
    assert fields[names.index("OVNGAP")] == "$open/Ref($close,1)-1"


def test_no_forward_refs():
    # No lookahead: every Ref uses a POSITIVE shift (past). A negative Ref
    # like Ref($x,-2) would peek into the future.
    import re
    fields, _ = short_term_factor_config()
    for f in fields:
        for m in re.findall(r"Ref\([^,]+,\s*(-?\d+)\)", f):
            assert int(m) > 0, f"forward Ref in {f!r}"
```

- [ ] **Step 2: 运行确认失败**
Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_short_term_factors.py -v`
Expected: FAIL (`No module named 'production.factors'`)

- [ ] **Step 3: 实现**
```python
# production/factors/__init__.py
```
```python
# production/factors/short_term.py
"""Non-redundant short-term alpha factors (pure qlib expressions, no extra data).
All trailing (Ref +k = past) -> no lookahead. See spec 2026-06-03-shortterm-factors."""
from __future__ import annotations


def short_term_factor_config() -> tuple[list[str], list[str]]:
    """Return (expression_fields, names) appended onto Alpha158 for LGBM."""
    fields = [
        "$open/Ref($close,1)-1",                          # overnight gap
        "Mean($open/Ref($close,1)-1, 5)",                 # 5d mean gap
        "Std($open/Ref($close,1)-1, 5)",                  # 5d gap vol
        "($vwap*$volume)/Mean($vwap*$volume, 20)",        # amount surge (money flow proxy)
        "Sum(Gt($close/Ref($close,1)-1, 0.095), 20)",     # ~10% limit-up touches / 20d
        "Sum(Gt($close/Ref($close,1)-1, 0.19), 20)",      # ~20% (ChiNext/STAR) touches / 20d
    ]
    names = ["OVNGAP", "OVNGAP_MA5", "OVNGAP_STD5",
             "AMT_SURGE", "LIMITUP10_CNT20", "LIMITUP20_CNT20"]
    return fields, names
```

- [ ] **Step 4: 运行确认通过**
Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_short_term_factors.py -v`
Expected: 3 passed

- [ ] **Step 5: 提交**
```bash
git add production/factors/__init__.py production/factors/short_term.py production/tests/test_short_term_factors.py
git commit -m "feat(factors): non-redundant short-term factor expressions"
```

---

### Task 2: AlphaShortTerm handler

**Files:** Modify `production/custom_handler.py`; Test append `production/tests/test_short_term_factors.py`

- [ ] **Step 1: 追加失败测试**
```python
# append to production/tests/test_short_term_factors.py
def test_alpha_short_term_appends_factors_to_alpha158():
    from custom_handler import AlphaShortTerm
    import sys
    from pathlib import Path
    # get_feature_config needs no data load; bypass __init__ via __new__.
    h = AlphaShortTerm.__new__(AlphaShortTerm)
    fields, names = h.get_feature_config()
    # superset of Alpha158 (158) + our 6
    assert "OVNGAP" in names and "LIMITUP10_CNT20" in names
    assert len(names) >= 158 + 6
```
(Note: `custom_handler` is importable because `production/` is on sys.path when tests run from repo root via the existing conftest/path setup; if the import fails in isolation, add `import sys; sys.path.insert(0, "production")` at the top of the test.)

- [ ] **Step 2: 运行确认失败**
Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_short_term_factors.py -k alpha_short_term -v`
Expected: FAIL (`cannot import name 'AlphaShortTerm'`)

- [ ] **Step 3: 实现** — append to `production/custom_handler.py`
```python
# append to production/custom_handler.py
from production.factors.short_term import short_term_factor_config


class AlphaShortTerm(Alpha158_OpenH):
    """Alpha158 (open-to-open label) + non-redundant short-term factors.

    Adds OVNGAP / AMT_SURGE / LIMITUP*_CNT20 etc. on top of the 158 features,
    for the LGBM tabular path. Neural factor injection is deferred (P3+).
    """

    def get_feature_config(self):
        fields, names = super().get_feature_config()
        extra_fields, extra_names = short_term_factor_config()
        return list(fields) + extra_fields, list(names) + extra_names
```
(If `from production.factors...` fails under the `custom_handler` import path used by qlib's config loader, use the path-robust form: `from factors.short_term import short_term_factor_config` — `production/` is already inserted into sys.path by `rolling_train` (line 226-228). Pick whichever import resolves; verify in Step 4.)

- [ ] **Step 4: 运行确认通过**
Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_short_term_factors.py -v`
Expected: 4 passed
Also confirm import both ways works: `F:/Tools/Anaconda/envs/qlib/python.exe -c "import sys; sys.path.insert(0,'production'); from custom_handler import AlphaShortTerm; print('ok')"`

- [ ] **Step 5: 提交**
```bash
git add production/custom_handler.py production/tests/test_short_term_factors.py
git commit -m "feat(factors): AlphaShortTerm handler (Alpha158 + short-term factors)"
```

---

### Task 3: rolling_train `--features` 选择

**Files:** Modify `production/rolling_train.py`; Test append `production/tests/test_short_term_factors.py`

- [ ] **Step 1: 追加失败测试**
```python
# append to production/tests/test_short_term_factors.py
def test_resolve_feature_handler():
    from production.rolling_train import resolve_feature_handler
    assert resolve_feature_handler("alpha158") == ("Alpha158_OpenH", "custom_handler")
    assert resolve_feature_handler("shortterm") == ("AlphaShortTerm", "custom_handler")
    import pytest
    with pytest.raises(ValueError):
        resolve_feature_handler("bogus")
```

- [ ] **Step 2: 运行确认失败**
Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_short_term_factors.py -k resolve_feature -v`
Expected: FAIL (`cannot import name 'resolve_feature_handler'`)

- [ ] **Step 3: 实现** — edit `production/rolling_train.py`

(a) Add the resolver near the top (after imports):
```python
def resolve_feature_handler(features: str) -> tuple[str, str]:
    """Map a --features choice to (handler_class_name, module_path)."""
    if features == "alpha158":
        return ("Alpha158_OpenH", "custom_handler")
    if features == "shortterm":
        return ("AlphaShortTerm", "custom_handler")
    raise ValueError(f"unknown features {features!r}")
```

(b) `train_lgbm_horizon(cfg, horizon, universe_name, end_date)` — add a keyword-only `features: str = "alpha158"` param. Replace the hardcoded handler import/instantiation (lines ~230-239) with class selection, and update `handler_cfg["class"]`:
```python
    from custom_handler import Alpha158_OpenH, AlphaShortTerm  # noqa: E402
    _handler_cls = {"Alpha158_OpenH": Alpha158_OpenH, "AlphaShortTerm": AlphaShortTerm}
    cls_name, module_path = resolve_feature_handler(features)
    handler = _handler_cls[cls_name](
        horizon_days=cfg.horizon_days[horizon.name],
        start_time=str(s.train_start),
        end_time=str(s.test_end),
        fit_start_time=str(s.train_start),
        fit_end_time=str(s.train_end),
        instruments=universe_name,
    )
```
and in `handler_cfg`, set `"class": cls_name` (instead of the literal `"Alpha158_OpenH"`).

(c) Thread `features` through callers: `run_once(cfg, end_date, ..., features: str = "alpha158")` passes `features` into its `train_lgbm_horizon(...)` call; `run_backfill(..., features: str = "alpha158")` passes it to `run_once`; the `backfill` and `run-once` CLI subparsers add `--features` (choices `["alpha158","shortterm"]`, default `"alpha158"`) and pass it through.

(All new params keyword-only with default `"alpha158"` → existing behavior unchanged.)

- [ ] **Step 4: 运行确认通过 + 无回归**
Run: `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_short_term_factors.py production/tests/test_backfill_schedule.py production/tests/test_rolling_train_pipeline.py -v`
Expected: all pass (new + existing rolling-train tests unchanged)

- [ ] **Step 5: 提交**
```bash
git add production/rolling_train.py production/tests/test_short_term_factors.py
git commit -m "feat(factors): --features {alpha158,shortterm} handler selection in rolling_train"
```

---

### Task 4（执行）: 因子 LGBM 长回填 + 增量评估

- [ ] **预检 + 看门狗**：`F:/Tools/Anaconda/envs/qlib/python.exe -m production.safety_watchdog > logs/watchdog.log 2>&1`（后台）。
- [ ] **冒烟（1 折，验证因子 handler 训练通）**：`F:/Tools/Anaconda/envs/qlib/python.exe -m production.rolling_train backfill --start 2024-12-27 --end 2024-12-27 --only-models lgbm --step-weeks 26 --test-weeks 26 --train-years 3 --features shortterm` —— 确认产出 `lgbm_*_2024-12-27` recorder、日志无异常、特征含新因子。
- [ ] **长回填（~2h，后台，LGBM-only）**：`F:/Tools/Anaconda/envs/qlib/python.exe -m production.rolling_train backfill --start 2021-01-01 --end 2026-01-01 --only-models lgbm --step-weeks 26 --test-weeks 26 --train-years 3 --features shortterm > logs/backfill_lgbmfac.log 2>&1`
  - ⚠️ 这会覆盖同 end_date 的 `lgbm_*` recorder（因子版替换基线版）。**先备份基线 OOF**：`oof_lgbm_2021_2026.pkl` 已存在于 reports，作为对照保留；如需保留基线 recorder，先 `--features alpha158` 的已在；用 pool 出的 pkl 对照即可（recorder 名相同会被覆盖，但 baseline 的 `oof_lgbm_2021_2026.pkl` 已落盘，对照不丢）。
- [ ] **拼 OOF**：`F:/Tools/Anaconda/envs/qlib/python.exe -m production.backfill_pool --start 2021-01-01 --end 2026-01-01 --models lgbm --out production/reports/oof_lgbmfac_2021_2026.pkl`
- [ ] **对比评估**（fixed/k5/p5/¥10万/小成本，复用 run.py/_t5 风格）：
  - LGBM+因子 vs LGBM-baseline：`run --pred-file oof_lgbmfac_2021_2026.pkl ...` 对比 `oof_lgbm_2021_2026.pkl` 的 net_cagr/net_ir/IC。
  - (LGBM+因子 + ALSTM)：把 `oof_lgbmfac` 与 alstm recorders 一起 `backfill_pool --models lgbm,alstm`（注意 lgbm recorder 已是因子版）→ 与现有 +19% 2 模型对比。
  - 叠加 P3 暴露层（`--regime trend_ma --ma-window 60`）看回撤/Calmar。
- [ ] **结果文档 + 回归 + 提交**：写 `docs/superpowers/specs/2026-06-03-shortterm-factors-results.md`（因子增量 delta、是否抬升、诚实结论含负结果）；跑全 backtest+factors 测试套件确认无回归；提交。

---

## Self-Review
**Spec 覆盖**：§2 因子集→T1；§3 集成(handler)→T2；handler 选择/回填→T3；§4 评估→T4。✅
**无前视**：因子全尾部表达式 + T1 `test_no_forward_refs` 断言。✅
**Placeholder**：T1–T3 完整代码+命令；T4 执行 runbook 命令明确。注意 T4 长回填会覆盖同名 lgbm recorder——已说明用已落盘的 `oof_lgbm_2021_2026.pkl` 作基线对照，不丢。
**类型一致**：`short_term_factor_config()->(fields,names)`（T1）；`AlphaShortTerm.get_feature_config`（T2）；`resolve_feature_handler(features)->(cls_name,module_path)` + `train_lgbm_horizon(...,features=)` + `run_once/run_backfill/CLI --features`（T3）一致；评估复用 backfill_pool/run.py。✅
**风险**：T2 的 import 路径（`production.factors` vs `factors`）依赖 sys.path——Step 4 显式双向验证。T3 改 rolling_train 生产函数——新参数默认 alpha158，向后兼容，跑现有 rolling-train 测试确认无回归。

## 后续
- 若因子有效：board-aware 精确涨停、真换手率(补 turn)、特异波动率；神经网络因子注入。
- 把最终模型(含有效因子+暴露层)落进产品。
