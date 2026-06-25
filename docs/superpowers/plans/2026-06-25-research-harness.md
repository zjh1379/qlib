# research-runner harness — 实现计划 (arch #2)

> REQUIRED SUB-SKILL: subagent-driven-development. Work in the CURRENT worktree dir (branch `claude/arch-deepening`), do NOT `cd`. Python `F:/Tools/Anaconda/envs/qlib/python.exe`.

**Goal:** `production/research/_harness.py` 收编 runner 样板;迁移 8 个 A2 期 runner。行为/数值不变。

---

## Task 1: 创建 `_harness.py` + 纯单测 (TDD)

- [ ] Step 1 写 `production/tests/test_harness.py`:

```python
import sys
import pytest
from production.research import _harness


def test_pct_formats_and_nan():
    assert _harness.pct(0.123) == "+12.3%"
    assert _harness.pct(-0.05) == "-5.0%"
    assert _harness.pct(float("nan")) == "n/a"
    assert _harness.pct(None) == "n/a"


def test_num_formats_and_nan():
    assert _harness.num(1.2345) == "1.23"
    assert _harness.num(1.2345, 4) == "1.2345"
    assert _harness.num(float("inf")) == "n/a"


def test_bootstrap_idempotent_inserts_purelib():
    import sysconfig
    _harness._BOOTSTRAPPED = False
    _harness.bootstrap()
    purelib = sysconfig.get_paths().get("purelib")
    assert purelib in sys.path
    assert _harness._BOOTSTRAPPED is True
    _harness.bootstrap()  # idempotent, no raise
```

- [ ] Step 2 run → FAIL. `F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_harness.py -q`
- [ ] Step 3 create `production/research/_harness.py`:

```python
# production/research/_harness.py
"""Shared boilerplate for production/research/ eval runners: env bootstrap, the
OOF -> champion-score contract, and table formatters — so each runner is just its
experiment, not five bands of copied header."""
from __future__ import annotations

import sys as _sys
import sysconfig as _sysconfig
from pathlib import Path

import numpy as np

OOF_FAC = "production/reports/oof_lgbmfac_2021_2026.pkl"
OOF_2MODEL = "production/reports/oof_2model_2021_2026.pkl"
CONFIG = "production/configs/rolling_ensemble.yaml"

_BOOTSTRAPPED = False


def bootstrap() -> None:
    """Idempotent: installed (compiled) qlib ahead of ./qlib source, utf-8 stdout,
    ensure logs/. Call at the top of a runner module, before any qlib import."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    purelib = _sysconfig.get_paths().get("purelib")
    if purelib and purelib not in _sys.path[:1]:
        _sys.path.insert(0, purelib)
    try:
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    Path("logs").mkdir(exist_ok=True)
    _BOOTSTRAPPED = True


def champion_scores():
    """Champion factor-2model score = rebuild_2model(OOF_FAC, OOF_2MODEL)."""
    import pandas as pd
    from production.score_utils import rebuild_2model
    return rebuild_2model(pd.read_pickle(OOF_FAC), pd.read_pickle(OOF_2MODEL))


def pct(x, nd: int = 1) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:+.{nd}%}"


def num(x, nd: int = 2) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"
```

- [ ] Step 4 run → PASS (3). Step 5 commit `feat(research): _harness (bootstrap + champion_scores + pct/num)`.

---

## Task 2: 迁移 8 个 A2 期 runner

**Recipe (每个 runner):** READ 它,然后:
1. 删掉模块顶部 `import sys as _sys, sysconfig as _sysconfig` + purelib fixup 块 + `try: _sys.stdout.reconfigure(...)` 块。
2. 在模块顶部(`import pandas` 等之前)加:
   ```python
   from production.research._harness import bootstrap, OOF_FAC, OOF_2MODEL, CONFIG
   bootstrap()
   ```
   (按需追加 `champion_scores`、`pct`、`num`。)
3. 删除本地 `OOF_FAC/OOF_2MODEL/CONFIG` 常量定义(改为上面 import)。
4. `Path("logs").mkdir(...)`(若在 main 内)可删(bootstrap 已建);删了不影响。
5. 见下表做 score-load 与 formatter 替换。**不动实验逻辑/输出格式以外的任何东西。**

| runner | score 替换 | formatter |
|--|--|--|
| `_eval_topk_sweep` | `factor_2m = champion_scores()`;保留 `base_2m = score_of(two)`(`two = pd.read_pickle(OOF_2MODEL)`);删本地 `_calmar`,改用 `from production.score_utils import calmar` | 删本地 `_pct/_num` → 用 harness `pct/num`(调用处 `_pct(`→`pct(`,`_num(`→`num(`) |
| `_eval_executability` | `scores = champion_scores()` | 无(inline f-string) |
| `_eval_deploy` | `scores = champion_scores()` | 无(保留其本地 `cal()`) |
| `_eval_user_exec` | `scores = champion_scores()` | 无(保留其 `_metrics`) |
| `_eval_am30_entry` | `scores = champion_scores()` | 无 |
| `_reconcile_live` | `scores = champion_scores()` | 无(保留 argparse/main) |
| `_eval_etf_timing` | **保留** `two = pd.read_pickle(OOF_2MODEL); s = score_of(two)`(只换头部+OOF常量来源) | 无 |
| `_eval_etf_real` | **保留** `score_of(two)` 同上 | 无 |

- [ ] 逐个改完后验证(**全部纯 import,不需数据**):
  ```
  F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_harness.py -q
  F:/Tools/Anaconda/envs/qlib/python.exe -c "import production.research._eval_topk_sweep, production.research._eval_executability, production.research._eval_deploy, production.research._eval_user_exec, production.research._eval_am30_entry, production.research._reconcile_live, production.research._eval_etf_timing, production.research._eval_etf_real; print('import OK')"
  ```
  两者都须通过(import OK 证明 8 个 runner 头部迁移没破)。
- [ ] commit `refactor(research): 8 A2 runners adopt _harness (boilerplate consolidated)`.

## 自评判据(控制者)
- `test_harness` 3 测过;8 runner import 全 OK;champion_scores 用于 6 个、score_of 保留于 2 个 ETF runner;`_eval_topk_sweep` 的 `_calmar/_pct/_num` 已换 canonical;无实验逻辑改动。控制者再跑 1-2 个 runner(worktree 有 OOF 缓存)确认端到端数值不变。
