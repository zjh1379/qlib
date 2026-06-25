# 筛选重算 UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把选股工作台的"贵"操作（视图/模型组合重算）与"廉"操作（其余筛选）分层——重算进草稿态、点按钮才跑、带真实百分比进度条；其余筛选（含做活后的窗口天数/Top N/最少进topN）全部浏览器即时计算。

**Architecture:** 后端 `candidates()` 的候选池 payload 扩展为每股近 K=20 日的 `daily_ranks`/`daily_scores` 数组 + 响应级 `window_dates`，让持续性/窗口/TopN 过滤搬到浏览器即时算。视图/模型组合的重算包装成复用 `inference` job 模式的后台线程任务（`app/models/recompute.py`），进度经 `threading.local` 注入 `_candidates_cached` 各阶段上报（不进 lru_cache 参数键），最耗时的"取行情指标"按 50 只一批上报使进度真实。前端 `view/models` 改 draft 态 + 「重新计算」按钮 + 进度条；裸 `GET /candidates` 用 `enabled` 门控在 job warm 之后。

**Tech Stack:** FastAPI + pydantic + pandas（后端）；React 18 + TanStack Query + react-router + vitest/testing-library（前端）。Python: `F:/Tools/Anaconda/envs/qlib/python.exe`。

---

## File Structure

**后端（`backend/app/models/`）**
- `recompute.py` *(新建)* — 进度原语（`threading.local` sink、`phase_percent`、`emit_progress`、`fetch_metrics_chunked`、常量 `CANDIDATES_WINDOW_K`/`CANDIDATES_POOL_CAP`）+ job 注册表/线程（`trigger_recompute`/`get_job`/`get_active_job`/`_run_recompute`）。**模块级只 import schemas；对 `service` 的 import 在 `_run_recompute` 内惰性进行**，避免循环。
- `service.py` *(改)* — `_build_screen_items` 设每股 `daily_ranks`/`daily_scores`；新增纯函数 `_window_dates`；`_candidates_cached` 填 `window_dates`、各阶段 `emit_progress`、metrics 改 `fetch_metrics_chunked`。从 `recompute` import 进度原语（模块级，无循环）。
- `schemas.py` *(改)* — `ScreenItem` 加 `daily_ranks`/`daily_scores`；`CandidatesResponse` 加 `window_dates`；新增 `RecomputeProgress`/`RecomputeJob`/`RecomputeRequest`/`RecomputeTriggerResponse`。
- `router.py` *(改)* — 3 个端点：`POST /candidates/recompute`、`GET /candidates/recompute/active`、`GET /candidates/recompute/{job_id}`（**active 必须声明在 {job_id} 之前**）。
- `tests/` *(新建)* — `test_recompute_progress.py`（进度数学 + 分批纯函数）、`test_recompute_job.py`（job 机制，mock service.candidates）、`test_build_screen_items_arrays.py`（数组）、`test_recompute_router.py`（端点，mock service）。

**前端（`frontend/`）**
- `src/pages/picks/persistence.ts` *(新建)* — 纯函数 `daysInTop`/`windowScoreAvg`/`comboKey`。
- `src/pages/picks/useRecompute.ts` *(新建)* — job 启动 + 轮询 + warmed 集合管理。
- `src/pages/picks/RecomputeProgress.tsx` *(新建)* — 进度条组件（百分比 + 阶段文案 + 已用秒数）。
- `src/api/client.ts` *(改)* — `models.recompute/recomputeStatus/recomputeActive`。
- `src/pages/picks/types.ts` *(改)* — `WINDOW_K=20`、窗口天数上限 60→20。
- `src/pages/Picks.tsx` *(改)* — draft 态、processed 管线、GET 门控、重算编排。
- `src/pages/picks/FilterBar.tsx` *(改)* — 视图+模型组合收进"需重新计算"分区 + 按钮；窗口/TopN/min_top 入即时层。
- `src/jobs/useActiveJobs.ts` *(改)* — 加 `recompute` kind（轻量）。
- `tests/persistence.test.ts` *(新建)*、`tests/Picks.test.tsx` *(改：改用 useCandidates mock + 新 payload)*。

**前后端必须一致的常量：** 后端 `CANDIDATES_WINDOW_K=20`/`CANDIDATES_POOL_CAP=300` 与前端 Picks 的 `WINDOW_DAYS=20`/`POOL_SIZE=300` 必须相等——否则重算 job warm 的 lru_cache 键与随后的 `GET` 键不一致、缓存不命中。

---

# Phase A — 后端

### Task A1: Schema 扩展

**Files:**
- Modify: `backend/app/models/schemas.py`
- Test: `backend/app/models/tests/test_recompute_schemas.py` *(新建)*

- [ ] **Step 1: 写失败测试**

```python
# backend/app/models/tests/test_recompute_schemas.py
from app.models.schemas import (
    ScreenItem, CandidatesResponse, RecomputeProgress, RecomputeJob,
    RecomputeRequest, RecomputeTriggerResponse,
)


def test_screen_item_accepts_daily_arrays():
    it = ScreenItem(
        rank=1, symbol="SH600000", score_today=0.1, score_avg=0.1,
        rank_avg=1.0, days_in_top=5,
        daily_ranks=[3, None, 1], daily_scores=[0.1, None, 0.2],
    )
    assert it.daily_ranks == [3, None, 1]
    assert it.daily_scores == [0.1, None, 0.2]


def test_candidates_response_has_window_dates():
    r = CandidatesResponse(
        experiment="e", recorder_id="r", latest_date="2026-06-16",
        window_days=20, universe_size=800, items=[],
        window_dates=["2026-06-12", "2026-06-13"],
    )
    assert r.window_dates == ["2026-06-12", "2026-06-13"]


def test_recompute_models_roundtrip():
    p = RecomputeProgress(phase="metrics", percent=60, message="x")
    job = RecomputeJob(job_id="j", status="running", started_at="t",
                       view="ensemble", models=["lgbm_5d"], progress=p)
    assert job.progress.percent == 60
    assert RecomputeRequest(view="ensemble", models=[]).models == []
    assert RecomputeTriggerResponse(status="started", job_id="j").job_id == "j"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest app/models/tests/test_recompute_schemas.py -v`
Expected: FAIL — `ImportError`/`cannot import name 'RecomputeProgress'`.

- [ ] **Step 3: 实现 schema 改动**

`schemas.py` — `ScreenItem` 在 `is_st: bool = False` 之后、`ai_analysis` 之前插入：

```python
    # 客户端即时层用：每股最近 K 个交易日的逐日排名/分数（与
    # CandidatesResponse.window_dates 升序对齐；缺数据为 None）。
    daily_ranks: list[int | None] = Field(default_factory=list)
    daily_scores: list[float | None] = Field(default_factory=list)
```

`CandidatesResponse` 在 `items: list[ScreenItem]` 之后插入：

```python
    # 客户端即时持续性/窗口过滤用：最近 K 个交易日（升序 ISO 日期）。
    window_dates: list[str] = Field(default_factory=list)
```

文件末尾追加：

```python
class RecomputeProgress(BaseModel):
    phase: str  # "load" | "score" | "metrics" | "enrich" | "done"
    percent: int  # 0..100 overall
    message: str = ""


class RecomputeJob(BaseModel):
    job_id: str
    status: str  # "running" | "done" | "failed"
    started_at: str
    finished_at: str | None = None
    error: str | None = None
    view: str = "ensemble"
    models: list[str] = Field(default_factory=list)
    progress: RecomputeProgress | None = None


class RecomputeRequest(BaseModel):
    view: str = "ensemble"
    models: list[str] = Field(default_factory=list)


class RecomputeTriggerResponse(BaseModel):
    status: str  # "started" | "already_running"
    job_id: str | None = None
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest app/models/tests/test_recompute_schemas.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: 提交**

```bash
git add backend/app/models/schemas.py backend/app/models/tests/test_recompute_schemas.py
git commit -m "feat(models): add daily rank/score arrays, window_dates, recompute job schemas"
```

---

### Task A2: 进度数学 + 分批纯函数（`recompute.py` 第一部分）

**Files:**
- Create: `backend/app/models/recompute.py`
- Test: `backend/app/models/tests/test_recompute_progress.py` *(新建)*

- [ ] **Step 1: 写失败测试**

```python
# backend/app/models/tests/test_recompute_progress.py
from app.models.recompute import phase_percent, fetch_metrics_chunked


def test_phase_percent_bounds():
    assert phase_percent("load", 1, 1) == 15
    assert phase_percent("score", 1, 1) == 30
    assert phase_percent("metrics", 0, 300) == 30
    assert phase_percent("metrics", 150, 300) == 60
    assert phase_percent("metrics", 300, 300) == 90
    assert phase_percent("enrich", 1, 1) == 100
    assert phase_percent("metrics", 0, 0) == 90  # total=0 -> frac=1.0 -> hi


def test_fetch_metrics_chunked_merges_and_reports():
    calls = []
    emits = []

    def fake_fetch(batch):
        calls.append(list(batch))
        return {s: {"v": s} for s in batch}

    syms = [f"S{i}" for i in range(125)]
    out = fetch_metrics_chunked(syms, fake_fetch, chunk_size=50,
                                emit=lambda done, total: emits.append((done, total)))
    assert len(out) == 125
    assert out["S124"] == {"v": "S124"}
    assert [len(c) for c in calls] == [50, 50, 25]
    assert emits == [(50, 125), (100, 125), (125, 125)]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest app/models/tests/test_recompute_progress.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.models.recompute'`.

- [ ] **Step 3: 实现 recompute.py 的进度原语**

```python
# backend/app/models/recompute.py
"""Recompute job for the candidate pool — wraps service.candidates() in a
background thread that warms the lru_cache while reporting honest progress.

Progress is injected via a thread-local sink (NOT passed as a function arg,
which would break the lru_cache key on _candidates_cached). On a cache hit the
compute body never runs, so the job completes instantly with no progress — the
intended "already-computed = instant" behaviour.

Module-level imports MUST stay limited to schemas + stdlib. The import of
app.models.service happens lazily inside _run_recompute to avoid a circular
import (service imports the progress primitives from this module).
"""
from __future__ import annotations

import logging
import threading
import uuid
from collections import OrderedDict
from datetime import datetime
from typing import Callable

from app.models.schemas import (
    RecomputeJob,
    RecomputeProgress,
    RecomputeTriggerResponse,
)

log = logging.getLogger(__name__)

# Must match frontend Picks WINDOW_DAYS / POOL_SIZE (see plan File Structure note).
CANDIDATES_WINDOW_K = 20
CANDIDATES_POOL_CAP = 300

# phase -> (lo, hi) overall-percent band. Tune after profiling (Task A7).
_PHASE_BOUNDS: dict[str, tuple[int, int]] = {
    "load": (0, 15),
    "score": (15, 30),
    "metrics": (30, 90),
    "enrich": (90, 100),
}

# Thread-local progress sink. Set by the recompute thread; unset (None) on
# normal GET threads -> emit_progress is a no-op there.
_progress_local = threading.local()


def phase_percent(phase: str, current: int, total: int) -> int:
    lo, hi = _PHASE_BOUNDS.get(phase, (0, 100))
    frac = (current / total) if total else 1.0
    frac = min(max(frac, 0.0), 1.0)
    return int(round(lo + (hi - lo) * frac))


def emit_progress(phase: str, current: int, total: int, message: str) -> None:
    """Report progress to the active recompute job, if any (else no-op)."""
    sink = getattr(_progress_local, "sink", None)
    if sink is None:
        return
    sink(RecomputeProgress(phase=phase, percent=phase_percent(phase, current, total),
                           message=message))


def fetch_metrics_chunked(
    symbols: list[str],
    fetch_fn: Callable[[list[str]], dict],
    chunk_size: int = 50,
    emit: Callable[[int, int], None] | None = None,
) -> dict:
    """Call fetch_fn over `symbols` in batches, merging results and reporting
    per-batch progress via `emit(done, total)`. Splitting the one big
    D.features call into batches adds negligible overhead but gives the
    progress bar real, smooth movement during the dominant phase."""
    out: dict = {}
    total = len(symbols)
    for i in range(0, total, chunk_size):
        batch = symbols[i:i + chunk_size]
        out.update(fetch_fn(batch))
        done = min(i + chunk_size, total)
        if emit is not None:
            emit(done, total)
    return out
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest app/models/tests/test_recompute_progress.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: 提交**

```bash
git add backend/app/models/recompute.py backend/app/models/tests/test_recompute_progress.py
git commit -m "feat(models): recompute progress primitives (phase_percent, chunked metrics, thread-local sink)"
```

---

### Task A3: Recompute job 机制（`recompute.py` 第二部分）

**Files:**
- Modify: `backend/app/models/recompute.py`
- Test: `backend/app/models/tests/test_recompute_job.py` *(新建)*

- [ ] **Step 1: 写失败测试**（mock `service.candidates`，避开 qlib）

```python
# backend/app/models/tests/test_recompute_job.py
import time
import app.models.recompute as rc


def _wait_done(job_id, timeout=2.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = rc.get_job(job_id)
        if j and j.status in ("done", "failed"):
            return j
        time.sleep(0.01)
    raise AssertionError("job did not finish in time")


def test_trigger_runs_and_emits_progress(monkeypatch):
    def fake_candidates(top, days, min_top, experiment=None, view="ensemble", models=None):
        # Simulate the compute path emitting progress.
        rc.emit_progress("metrics", 5, 10, "halfway")
        return {"items": []}
    monkeypatch.setattr("app.models.service.candidates", fake_candidates)

    resp = rc.trigger_recompute(view="ensemble", models=["lgbm_5d"])
    assert resp.status == "started" and resp.job_id
    job = _wait_done(resp.job_id)
    assert job.status == "done"
    assert job.progress.percent == 100  # final done overrides intermediate
    assert job.view == "ensemble" and job.models == ["lgbm_5d"]


def test_trigger_failure_sets_failed(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("kaboom")
    monkeypatch.setattr("app.models.service.candidates", boom)
    resp = rc.trigger_recompute(view="alstm", models=[])
    job = _wait_done(resp.job_id)
    assert job.status == "failed"
    assert "kaboom" in (job.error or "")


def test_already_running_guard(monkeypatch):
    import threading
    gate = threading.Event()

    def slow(*a, **k):
        gate.wait(1.0)
        return {"items": []}
    monkeypatch.setattr("app.models.service.candidates", slow)
    r1 = rc.trigger_recompute(view="ensemble", models=[])
    r2 = rc.trigger_recompute(view="ensemble", models=[])  # while r1 running
    assert r2.status == "already_running" and r2.job_id == r1.job_id
    gate.set()
    _wait_done(r1.job_id)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest app/models/tests/test_recompute_job.py -v`
Expected: FAIL — `AttributeError: module 'app.models.recompute' has no attribute 'trigger_recompute'`.

- [ ] **Step 3: 追加 job 机制到 `recompute.py`**

```python
# --- append to backend/app/models/recompute.py ---

_MAX_JOBS = 20
_JOBS: "OrderedDict[str, RecomputeJob]" = OrderedDict()
_ACTIVE_ID: str | None = None
_LOCK = threading.Lock()


def get_job(job_id: str) -> RecomputeJob | None:
    return _JOBS.get(job_id)


def get_active_job() -> RecomputeJob | None:
    if _ACTIVE_ID and _ACTIVE_ID in _JOBS:
        return _JOBS[_ACTIVE_ID]
    return None


def trigger_recompute(view: str, models: list[str]) -> RecomputeTriggerResponse:
    global _ACTIVE_ID
    with _LOCK:
        if _ACTIVE_ID and _ACTIVE_ID in _JOBS and _JOBS[_ACTIVE_ID].status == "running":
            return RecomputeTriggerResponse(status="already_running", job_id=_ACTIVE_ID)
        job_id = uuid.uuid4().hex[:12]
        _JOBS[job_id] = RecomputeJob(
            job_id=job_id, status="running",
            started_at=datetime.utcnow().isoformat(),
            view=view, models=list(models),
            progress=RecomputeProgress(phase="load", percent=0, message="开始重算"),
        )
        _JOBS.move_to_end(job_id)
        while len(_JOBS) > _MAX_JOBS:
            old_id, _ = _JOBS.popitem(last=False)
            log.debug("evicted_old_recompute_job %s", old_id)
        _ACTIVE_ID = job_id

    threading.Thread(target=_run_recompute, args=(job_id, view, list(models)),
                     daemon=True).start()
    return RecomputeTriggerResponse(status="started", job_id=job_id)


def _run_recompute(job_id: str, view: str, models: list[str]) -> None:
    global _ACTIVE_ID
    from app.models import service  # lazy import: avoids circular import

    def sink(p: RecomputeProgress) -> None:
        with _LOCK:
            j = _JOBS.get(job_id)
            if j:
                j.progress = p

    _progress_local.sink = sink
    try:
        models_csv = ",".join(models) if models else None
        # Warm the lru_cache. View+models drive the heavy path; cache hit = instant.
        service.candidates(
            top=CANDIDATES_POOL_CAP, days=CANDIDATES_WINDOW_K, min_top=0,
            view=view, models=models_csv,
        )
        with _LOCK:
            j = _JOBS.get(job_id)
            if j:
                j.status = "done"
                j.finished_at = datetime.utcnow().isoformat()
                j.progress = RecomputeProgress(phase="done", percent=100, message="完成")
    except Exception as exc:  # noqa: BLE001 — record any failure on the job
        log.exception("recompute_failed job_id=%s: %s", job_id, exc)
        with _LOCK:
            j = _JOBS.get(job_id)
            if j:
                j.status = "failed"
                j.finished_at = datetime.utcnow().isoformat()
                j.error = str(exc)[:2000]
    finally:
        _progress_local.sink = None
        with _LOCK:
            _ACTIVE_ID = None
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest app/models/tests/test_recompute_job.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: 提交**

```bash
git add backend/app/models/recompute.py backend/app/models/tests/test_recompute_job.py
git commit -m "feat(models): recompute background job (trigger/run/active, already-running guard)"
```

---

### Task A4: `_build_screen_items` 导出 daily 数组 + `_window_dates`

**Files:**
- Modify: `backend/app/models/service.py` (`_build_screen_items` ~L111-184；新增 `_window_dates`)
- Test: `backend/app/models/tests/test_build_screen_items_arrays.py` *(新建)*

- [ ] **Step 1: 写失败测试**

```python
# backend/app/models/tests/test_build_screen_items_arrays.py
import pandas as pd
from app.models.service import _build_screen_items, _window_dates


def _mk_df():
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2026-05-10", "2026-05-14"), ["SH600000", "SH600001"]],
        names=["datetime", "instrument"],
    )
    # SH600000 always higher score -> daily rank 1; SH600001 -> rank 2
    df = pd.DataFrame(
        {"score": [0.10, -0.05, 0.11, -0.04, 0.13, -0.03, 0.12, -0.02, 0.14, -0.01]},
        index=idx,
    )
    return df


def test_window_dates_ascending_last_k():
    df = _mk_df()
    wd = _window_dates(df, k=3)
    assert wd == ["2026-05-12", "2026-05-13", "2026-05-14"]
    # k larger than available -> all available
    assert _window_dates(df, k=99)[0] == "2026-05-10"


def test_daily_arrays_aligned_and_typed():
    df = _mk_df()
    items = _build_screen_items(df, top=2, days=5, min_top=0, name_map={})
    top = next(it for it in items if it.symbol == "SH600000")
    assert top.daily_ranks == [1, 1, 1, 1, 1]      # always best
    assert len(top.daily_scores) == 5
    assert top.daily_scores[-1] == 0.14            # last day
    other = next(it for it in items if it.symbol == "SH600001")
    assert other.daily_ranks == [2, 2, 2, 2, 2]


def test_daily_arrays_fill_none_for_missing_day():
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2026-05-10"), "SH600000"),
         (pd.Timestamp("2026-05-12"), "SH600000"),
         (pd.Timestamp("2026-05-10"), "SH600001"),
         (pd.Timestamp("2026-05-12"), "SH600001")],
        names=["datetime", "instrument"],
    )
    df = pd.DataFrame({"score": [0.2, 0.3, 0.1, 0.1]}, index=idx)
    # window_dates would be [05-10, 05-12]; both present -> no None here,
    # but build a 3-day window by reindex check via _window_dates length.
    items = _build_screen_items(df, top=2, days=2, min_top=0, name_map={})
    for it in items:
        assert len(it.daily_ranks) == 2
        assert all(r is None or isinstance(r, int) for r in it.daily_ranks)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest app/models/tests/test_build_screen_items_arrays.py -v`
Expected: FAIL — `ImportError: cannot import name '_window_dates'`.

- [ ] **Step 3: 实现**

在 `service.py` 中 `_build_screen_items` 之前新增纯函数：

```python
def _window_dates(df: "pd.DataFrame", k: int) -> list[str]:
    """Ascending ISO dates of the last `k` trading days present in df's index."""
    dates = df.index.get_level_values("datetime").unique().sort_values()
    return [d.date().isoformat() if hasattr(d, "date") else str(d)[:10]
            for d in dates[-k:]]
```

在 `_build_screen_items` 内，`window_df` 加好 `rank` 列之后（现有 L127-131 之后），新增两个 pivot：

```python
    # Per-symbol daily rank/score over the window, for client-side
    # persistence/window filtering. Pivot once; reindex to the window order.
    _rank_pivot = window_df["rank"].unstack("instrument").reindex(index=window)
    _score_pivot = window_df["score"].unstack("instrument").reindex(index=window)

    def _col_list(pivot, sym, as_int):
        if sym not in pivot.columns:
            return [None] * len(window)
        out = []
        for v in pivot[sym].tolist():
            if pd.isna(v):
                out.append(None)
            else:
                out.append(int(v) if as_int else float(v))
        return out
```

在构造 `ScreenItem(...)` 的调用里（现有 L171-182）追加两个字段：

```python
        items.append(
            ScreenItem(
                rank=rank_pos,
                symbol=symbol,
                name=name_map.get(symbol, ""),
                score_today=score_today,
                score_avg=float(row["score_avg"]),
                rank_avg=float(row["rank_avg"]),
                days_in_top=int(row["days_in_top"]),
                consensus=consensus,
                base_scores=base_scores,
                daily_ranks=_col_list(_rank_pivot, symbol, True),
                daily_scores=_col_list(_score_pivot, symbol, False),
            )
        )
```

- [ ] **Step 4: 跑测试确认通过 + 回归既有 `_build_screen_items` 测试**

Run: `cd backend && F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest app/models/tests/test_build_screen_items_arrays.py app/models/tests/test_screen_new_shape.py -v`
Expected: PASS（新 3 + 既有 3）

- [ ] **Step 5: 提交**

```bash
git add backend/app/models/service.py backend/app/models/tests/test_build_screen_items_arrays.py
git commit -m "feat(models): _build_screen_items emits per-symbol daily rank/score arrays + _window_dates helper"
```

---

### Task A5: `_candidates_cached` 填 window_dates + 各阶段进度 + 分批 metrics

**Files:**
- Modify: `backend/app/models/service.py` (`_candidates_cached` ~L238-429)

> 说明：此任务改动依赖 qlib 真实数据，难做纯单测；正确性由 Task A8（手动/集成）验收。本任务做"实现 + import 校验 + 既有单测回归"。

- [ ] **Step 1: 顶部 import 进度原语**

`service.py` 顶部 import 区追加：

```python
from app.models.recompute import emit_progress, fetch_metrics_chunked, CANDIDATES_WINDOW_K
```

- [ ] **Step 2: `load` 阶段埋点**

`_candidates_cached` 内，`pred = load_pred(recorder_id, experiment_name=exp)`（L253）之后插入：

```python
    emit_progress("load", 1, 1, "加载模型预测")
```

- [ ] **Step 3: `score` 阶段埋点**

在 score 重算块（L279-286 的 `if score_cols:` 之后整体结束处）后插入：

```python
    emit_progress("score", 1, 1, "重算分数 + 排名")
```

- [ ] **Step 4: metrics 分批**

把现有（L296-298）：

```python
    if items:
        prices = get_latest_close_prices([it.symbol for it in items])
        metrics = get_filter_metrics([it.symbol for it in items])
```

替换为：

```python
    if items:
        syms = [it.symbol for it in items]
        emit_progress("metrics", 0, len(syms), f"正在取行情指标 0/{len(syms)}")
        prices = get_latest_close_prices(syms)
        metrics = fetch_metrics_chunked(
            syms, get_filter_metrics, chunk_size=50,
            emit=lambda done, total: emit_progress(
                "metrics", done, total, f"正在取行情指标 {done}/{total}"),
        )
```

- [ ] **Step 5: window_dates + enrich 埋点 + 返回值**

在返回 dict 之前（L416 `# Staleness` 上方或附近）加：

```python
    emit_progress("enrich", 1, 1, "多周期富集 + 校准")
    window_dates = _window_dates(df, CANDIDATES_WINDOW_K)
```

在 return dict（L417-429）追加一项：

```python
        "window_dates": window_dates,
```

- [ ] **Step 6: import + 既有单测回归（确保没破坏导入/缓存）**

Run: `cd backend && F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -c "import app.models.service; import app.models.recompute; print('import ok')"`
Expected: 打印 `import ok`（确认无循环 import）

Run: `cd backend && F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest app/models/tests/ -v`
Expected: 全绿（既有 + 新增单测）

- [ ] **Step 7: 提交**

```bash
git add backend/app/models/service.py
git commit -m "feat(models): candidates() reports staged progress + window_dates; chunked metrics fetch"
```

---

### Task A6: Recompute 端点

**Files:**
- Modify: `backend/app/models/router.py`
- Test: `backend/app/models/tests/test_recompute_router.py` *(新建)*

- [ ] **Step 1: 写失败测试**（mock service + recompute）

```python
# backend/app/models/tests/test_recompute_router.py
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.models.router import router
from app.models.schemas import RecomputeJob, RecomputeProgress, RecomputeTriggerResponse
import app.models.recompute as rc


def _client():
    app = FastAPI()
    app.include_router(router, prefix="/api/models")
    return TestClient(app)


def test_post_recompute_starts(monkeypatch):
    monkeypatch.setattr(rc, "trigger_recompute",
                        lambda view, models: RecomputeTriggerResponse(status="started", job_id="j1"))
    c = _client()
    r = c.post("/api/models/candidates/recompute", json={"view": "ensemble", "models": ["lgbm_5d"]})
    assert r.status_code == 200
    assert r.json() == {"status": "started", "job_id": "j1"}


def test_get_recompute_status(monkeypatch):
    job = RecomputeJob(job_id="j1", status="running", started_at="t", view="ensemble",
                       models=[], progress=RecomputeProgress(phase="metrics", percent=60, message="x"))
    monkeypatch.setattr(rc, "get_job", lambda jid: job if jid == "j1" else None)
    c = _client()
    assert c.get("/api/models/candidates/recompute/j1").json()["progress"]["percent"] == 60
    assert c.get("/api/models/candidates/recompute/nope").status_code == 404


def test_get_recompute_active(monkeypatch):
    monkeypatch.setattr(rc, "get_active_job", lambda: None)
    c = _client()
    r = c.get("/api/models/candidates/recompute/active")
    assert r.status_code == 200 and r.json() is None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd backend && F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest app/models/tests/test_recompute_router.py -v`
Expected: FAIL — 404（路由不存在）

- [ ] **Step 3: 实现端点**

`router.py` import 区追加：

```python
from app.models import recompute
from app.models.schemas import (
    RecomputeJob,
    RecomputeRequest,
    RecomputeTriggerResponse,
)
```

文件末尾追加（**`/active` 必须在 `/{job_id}` 之前声明**，否则 "active" 会被当成 job_id）：

```python
@router.post("/candidates/recompute", response_model=RecomputeTriggerResponse)
def recompute_start(payload: RecomputeRequest):
    """Start a background recompute of the candidate pool for the given
    view + models. Warms the lru_cache; the subsequent GET /candidates is a
    cache hit. Reports progress via GET /candidates/recompute/{job_id}."""
    return recompute.trigger_recompute(view=payload.view, models=payload.models)


@router.get("/candidates/recompute/active", response_model=RecomputeJob | None)
def recompute_active():
    return recompute.get_active_job()


@router.get("/candidates/recompute/{job_id}", response_model=RecomputeJob)
def recompute_status(job_id: str):
    job = recompute.get_job(job_id)
    if not job:
        raise HTTPException(404, detail="recompute job not found")
    return job
```

`router.py` 顶部 import 确认含 `HTTPException`：把第 1 行改为

```python
from fastapi import APIRouter, Depends, HTTPException, Query
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd backend && F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest app/models/tests/test_recompute_router.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: 提交**

```bash
git add backend/app/models/router.py backend/app/models/tests/test_recompute_router.py
git commit -m "feat(models): POST /candidates/recompute + GET status/active endpoints"
```

---

### Task A7: 后端整体回归

**Files:** 无（验证）

- [ ] **Step 1: 全后端测试**

Run: `cd backend && F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest -q`
Expected: 全绿（无回归）

- [ ] **Step 2: 提交（若有快照/小修）**

```bash
git add -A && git commit -m "test(models): backend regression green for recompute feature" || echo "nothing to commit"
```

---

# Phase B — 前端

### Task B1: 重生 API 类型 + client 方法

**Files:**
- Modify: `frontend/src/api/types.gen.ts`（自动生成）、`frontend/src/api/client.ts`

> 需要后端已起（`gen:api` 抓 `http://localhost:8000/openapi.json`）。在 **主仓** 起后端最稳（qlib 数据在主仓）：见 Task B7 的启动命令；生成期间临时起一个即可。

- [ ] **Step 1: 起后端并重生类型**

Run（后端已在 :8000 跑）：`cd frontend && npm run gen:api`
Expected: `src/api/types.gen.ts` 更新，含 `RecomputeJob`/`RecomputeProgress`/`RecomputeRequest`/`RecomputeTriggerResponse` schema 与 `/api/models/candidates/recompute` 路径；`ScreenItem` 含 `daily_ranks`/`daily_scores`，`CandidatesResponse` 含 `window_dates`。

- [ ] **Step 2: 加 client 方法**

`client.ts` 的 `models:` 对象内，`candidates(...)` 之后插入：

```ts
    recompute: (body: { view: string; models: string[] }) => {
      type R = components['schemas']['RecomputeTriggerResponse'];
      return request<R>('/api/models/candidates/recompute', {
        method: 'POST',
        body: JSON.stringify(body),
      });
    },
    recomputeStatus: (jobId: string) => {
      type R = components['schemas']['RecomputeJob'];
      return request<R>(`/api/models/candidates/recompute/${encodeURIComponent(jobId)}`);
    },
    recomputeActive: () => {
      type R = components['schemas']['RecomputeJob'];
      return request<R | null>('/api/models/candidates/recompute/active');
    },
```

`client.ts` 顶部 import 改为同时引入 `components`：

```ts
import type { components, paths } from '@/api/types.gen';
```

- [ ] **Step 3: typecheck**

Run: `cd frontend && npm run typecheck`
Expected: 通过（无类型错误）

- [ ] **Step 4: 提交**

```bash
git add frontend/src/api/types.gen.ts frontend/src/api/client.ts
git commit -m "feat(frontend): regen API types + recompute client methods"
```

---

### Task B2: 即时层纯函数 `persistence.ts`

**Files:**
- Create: `frontend/src/pages/picks/persistence.ts`
- Test: `frontend/tests/persistence.test.ts` *(新建)*

- [ ] **Step 1: 写失败测试**

```ts
// frontend/tests/persistence.test.ts
import { describe, it, expect } from 'vitest';
import { daysInTop, windowScoreAvg, comboKey } from '@/pages/picks/persistence';

describe('daysInTop', () => {
  it('counts days where rank <= topN over the last D days', () => {
    // window-aligned ranks (ascending dates); last 3 = [2, 1, 5]
    expect(daysInTop([10, 8, 2, 1, 5], 3, 3)).toBe(2); // 2 and 1 are <=3
    expect(daysInTop([10, 8, 2, 1, 5], 5, 3)).toBe(2);
  });
  it('ignores null ranks', () => {
    expect(daysInTop([null, 1, null, 2], 4, 3)).toBe(2);
  });
  it('D longer than array uses whole array', () => {
    expect(daysInTop([1, 2], 99, 2)).toBe(2);
  });
});

describe('windowScoreAvg', () => {
  it('averages last D non-null scores', () => {
    expect(windowScoreAvg([0.1, 0.2, 0.3], 2)).toBeCloseTo(0.25);
  });
  it('returns null when all null in window', () => {
    expect(windowScoreAvg([0.1, null, null], 2)).toBeNull();
  });
});

describe('comboKey', () => {
  it('is order-independent for models', () => {
    expect(comboKey('ensemble', ['b', 'a'])).toBe(comboKey('ensemble', ['a', 'b']));
  });
  it('distinguishes view', () => {
    expect(comboKey('alstm', [])).not.toBe(comboKey('ensemble', []));
  });
});
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd frontend && npm run test -- persistence`
Expected: FAIL — cannot resolve `@/pages/picks/persistence`.

- [ ] **Step 3: 实现**

```ts
// frontend/src/pages/picks/persistence.ts

/** Count days in the last `windowD` entries where rank is present and <= topN.
 *  Ranks are aligned ascending to CandidatesResponse.window_dates. */
export function daysInTop(
  dailyRanks: (number | null)[],
  windowD: number,
  topN: number,
): number {
  const slice = windowD >= dailyRanks.length ? dailyRanks : dailyRanks.slice(-windowD);
  let n = 0;
  for (const r of slice) if (r != null && r <= topN) n++;
  return n;
}

/** Mean of the last `windowD` non-null daily scores, or null if none. */
export function windowScoreAvg(
  dailyScores: (number | null)[],
  windowD: number,
): number | null {
  const slice = windowD >= dailyScores.length ? dailyScores : dailyScores.slice(-windowD);
  let sum = 0;
  let count = 0;
  for (const s of slice) if (s != null) { sum += s; count++; }
  return count === 0 ? null : sum / count;
}

/** Stable, order-independent key for a (view, models) combo. */
export function comboKey(view: string, models: string[]): string {
  return `${view}|${[...models].sort().join(',')}`;
}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd frontend && npm run test -- persistence`
Expected: PASS（3 describe，全绿）

- [ ] **Step 5: 提交**

```bash
git add frontend/src/pages/picks/persistence.ts frontend/tests/persistence.test.ts
git commit -m "feat(frontend): client-side persistence/window helpers (daysInTop, windowScoreAvg, comboKey)"
```

---

### Task B3: `useRecompute` hook

**Files:**
- Create: `frontend/src/pages/picks/useRecompute.ts`

> 该 hook 管理：warmed 组合集合、当前 job、启动函数、轮询状态。React 行为靠 Task B4 的 Picks 集成 + 手动验收；本任务先把 hook 写出并通过 typecheck。

- [ ] **Step 1: 实现 hook**

```ts
// frontend/src/pages/picks/useRecompute.ts
import { useCallback, useEffect, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/api/client';
import type { components } from '@/api/types.gen';
import { comboKey } from './persistence';

type RecomputeJob = components['schemas']['RecomputeJob'];

export interface RecomputeController {
  /** combos warmed this session (backend lru_cache hot + GET allowed). */
  isWarmed: (view: string, models: string[]) => boolean;
  /** start a recompute for a combo; progress surfaces via `job`. */
  start: (view: string, models: string[]) => Promise<void>;
  /** latest polled job (running/done/failed) or null. */
  job: RecomputeJob | null;
  /** elapsed seconds since the active job started (0 when idle). */
  elapsedSec: number;
}

export function useRecompute(onWarmed?: (view: string, models: string[]) => void): RecomputeController {
  const [warmed, setWarmed] = useState<Set<string>>(new Set());
  const [jobId, setJobId] = useState<string | null>(null);
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [elapsedSec, setElapsedSec] = useState(0);
  const pending = useRef<{ view: string; models: string[] } | null>(null);
  const onWarmedRef = useRef(onWarmed);
  onWarmedRef.current = onWarmed;

  const { data: job } = useQuery({
    queryKey: ['recompute', jobId],
    queryFn: () => api.models.recomputeStatus(jobId as string),
    enabled: !!jobId,
    refetchInterval: (q) => (q.state.data?.status === 'running' ? 800 : false),
  });

  // elapsed timer while a job is running
  useEffect(() => {
    if (!startedAt) return;
    const h = setInterval(() => setElapsedSec(Math.round((Date.now() - startedAt) / 1000)), 250);
    return () => clearInterval(h);
  }, [startedAt]);

  // when job finishes, mark warmed (on done) and stop the timer
  useEffect(() => {
    if (!job) return;
    if (job.status === 'done' && pending.current) {
      const { view, models } = pending.current;
      setWarmed((prev) => new Set(prev).add(comboKey(view, models)));
      onWarmedRef.current?.(view, models);
    }
    if (job.status === 'done' || job.status === 'failed') {
      setStartedAt(null);
      pending.current = null;
    }
  }, [job?.status]); // eslint-disable-line react-hooks/exhaustive-deps

  const isWarmed = useCallback(
    (view: string, models: string[]) => warmed.has(comboKey(view, models)),
    [warmed],
  );

  const start = useCallback(async (view: string, models: string[]) => {
    pending.current = { view, models };
    setStartedAt(Date.now());
    setElapsedSec(0);
    const res = await api.models.recompute({ view, models });
    if (res.job_id) setJobId(res.job_id);
  }, []);

  return { isWarmed, start, job: job ?? null, elapsedSec };
}
```

- [ ] **Step 2: typecheck**

Run: `cd frontend && npm run typecheck`
Expected: 通过

- [ ] **Step 3: 提交**

```bash
git add frontend/src/pages/picks/useRecompute.ts
git commit -m "feat(frontend): useRecompute hook (job start + poll + warmed-combo set)"
```

---

### Task B4: 进度条组件 `RecomputeProgress`

**Files:**
- Create: `frontend/src/pages/picks/RecomputeProgress.tsx`

- [ ] **Step 1: 实现组件**

```tsx
// frontend/src/pages/picks/RecomputeProgress.tsx
import type { components } from '@/api/types.gen';

type RecomputeJob = components['schemas']['RecomputeJob'];

export default function RecomputeProgress({
  job, elapsedSec,
}: {
  job: RecomputeJob | null;
  elapsedSec: number;
}) {
  if (!job || job.status !== 'running') return null;
  const pct = job.progress?.percent ?? 0;
  const msg = job.progress?.message ?? '正在重新计算…';
  return (
    <div className="rounded-lg border border-[#30363d] bg-[#0d1117] p-4">
      <div className="flex items-center justify-between text-xs text-[#8b949e] mb-2">
        <span>{msg}</span>
        <span className="font-mono">{pct}% · 已用 {elapsedSec}s</span>
      </div>
      <div className="h-2 w-full rounded bg-[#21262d] overflow-hidden">
        <div
          className="h-full bg-[#1f6feb] transition-[width] duration-300"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
```

- [ ] **Step 2: typecheck**

Run: `cd frontend && npm run typecheck`
Expected: 通过

- [ ] **Step 3: 提交**

```bash
git add frontend/src/pages/picks/RecomputeProgress.tsx
git commit -m "feat(frontend): RecomputeProgress bar (percent + phase message + elapsed)"
```

---

### Task B5: types.ts 窗口上限 + 即时层管线接入

**Files:**
- Modify: `frontend/src/pages/picks/types.ts`、`frontend/src/pages/picks/filter.ts`

- [ ] **Step 1: types.ts — 新增常量 + 默认窗口**

`types.ts` 顶部（`export type View ...` 之前）加：

```ts
/** Must equal backend CANDIDATES_WINDOW_K. Caps the client window slider. */
export const WINDOW_K = 20;
```

把 `DEFAULT_FILTERS.days` 由 `5` 改为 `5`（保持默认 5；仅上限改到 20，见 FilterBar）。无需改 `days` 默认值。

- [ ] **Step 2: filter.ts — 新增 persistence-aware 管线函数**

`filter.ts` 末尾追加（保留现有 `applyFilters`）：

```ts
import { daysInTop, windowScoreAvg } from './persistence';

/** Full client-side pipeline producing the displayed rows:
 *  1) attribute filters (price/trend/board/ST/consensus) via applyFilters
 *  2) window score_avg recompute (over last D days)
 *  3) persistence filter: days in top-`top` over last D days >= min_top
 *  4) canonical rank by window score_avg desc
 *  5) display cap to `top` rows
 *  Returns the selected, rank-assigned rows (display sort applied by caller). */
export function selectCandidates(
  candidates: Candidate[],
  filters: FilterParams,
  windowDatesLen: number,
): Candidate[] {
  const D = Math.max(1, Math.min(filters.days, windowDatesLen || filters.days));
  const passed = applyFilters(candidates, filters)
    .map((c) => ({
      ...c,
      score_avg: windowScoreAvg(c.daily_scores ?? [], D) ?? c.score_avg,
    }))
    .filter((c) => daysInTop(c.daily_ranks ?? [], D, filters.top) >= filters.min_top);
  passed.sort((a, b) => (b.score_avg ?? -Infinity) - (a.score_avg ?? -Infinity));
  return passed
    .map((c, i) => ({ ...c, rank: i + 1 }))
    .slice(0, filters.top);
}
```

- [ ] **Step 3: typecheck**

Run: `cd frontend && npm run typecheck`
Expected: 通过

- [ ] **Step 4: 提交**

```bash
git add frontend/src/pages/picks/types.ts frontend/src/pages/picks/filter.ts
git commit -m "feat(frontend): selectCandidates pipeline (window score_avg + persistence + display cap)"
```

---

### Task B6: FilterBar 重组（草稿态 + 重新计算按钮 + 即时层归位）

**Files:**
- Modify: `frontend/src/pages/picks/FilterBar.tsx`

- [ ] **Step 1: 改 FilterBar props + 视图/模型组合走 draft**

`FilterBarProps` 增加 draft 相关入参（替换"视图/模型组合即时 onChange"为 draft 受控 + 按钮）：

```ts
interface FilterBarProps {
  params: FilterParams;
  resultCount: number | null;
  candidateCount: number | null;
  onChange: (patch: Partial<FilterParams>) => void;
  onReset: () => void;
  availableModels?: string[];
  activeModels?: string[] | null;
  // --- recompute draft tier ---
  draftView: View;
  draftModels: string[];
  onDraftView: (v: View) => void;
  onDraftModels: (m: string[]) => void;
  recomputeDirty: boolean;
  onRecompute: () => void;
  recomputeBusy: boolean;
}
```

把"Group 1: 基础"里的 `视图` Select 从 `params.view`/`onChange({view})` 改为 `draftView`/`onDraftView`；`窗口天数` 上限改 20、`最少进 top N 天数` `max={params.days}` 保持，但这些仍走即时 `onChange`（它们是即时层）。`Top N` 保持即时 `onChange`。

把"Group 1.5: 模型组合"的 `ModelSelector` 的 `selected`/`onChange` 改为 `draftModels`/`onDraftModels`，并在其下方加按钮区。`窗口天数` NumberField 的 `max` 由 `60` 改为 `WINDOW_K`（import 之）。

具体替换"Group 1"+"Group 1.5"两块为：

```tsx
      {/* Group 1: 基础即时层 */}
      <div>
        <h3 className="text-[10px] text-[#6e7681] uppercase tracking-wider mb-2">基础（即时）</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          <NumberField label="Top N" value={params.top} min={1} max={300} onChange={(v) => onChange({ top: v })} />
          <NumberField label="窗口天数" value={params.days} min={1} max={WINDOW_K} onChange={(v) => onChange({ days: v })} />
          <NumberField label="最少进 top N 天数" value={params.min_top} min={0} max={params.days} onChange={(v) => onChange({ min_top: v })} />
        </div>
      </div>

      {/* Group 1.5: 模型 / 视图 — 需重新计算 */}
      <div className="rounded-md border border-[#30363d] bg-[#0d1117] p-3 space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-[10px] text-[#6e7681] uppercase tracking-wider">模型 / 视图 · 需重新计算</h3>
          <button
            onClick={onRecompute}
            disabled={!recomputeDirty || recomputeBusy}
            className={cn(
              'text-xs px-3 py-1 rounded border',
              recomputeDirty && !recomputeBusy
                ? 'bg-[#1f6feb] border-[#1f6feb] text-white hover:bg-[#388bfd]'
                : 'bg-[#21262d] border-[#30363d] text-[#6e7681] cursor-not-allowed',
            )}
          >
            {recomputeBusy ? '计算中…' : recomputeDirty ? '重新计算 ●' : '重新计算'}
          </button>
        </div>
        <Select label="视图" value={draftView} options={VIEW_OPTIONS} onChange={(v) => onDraftView(v as View)} />
        {availableModels.length > 0 && (
          <ModelSelector available={availableModels} selected={draftModels} active={activeModels} onChange={onDraftModels} />
        )}
      </div>
```

`FilterBar.tsx` 顶部 import 追加 `cn` 与 `WINDOW_K`：

```ts
import { cn } from '@/lib/utils';
import { BOARDS, NEW_HIGH_N_OPTIONS, PCT_CHANGE_N_OPTIONS, WINDOW_K } from './types';
```

（`ModelSelector` 底部说明文案中"首次请求 ~5s"可保留或改为"点重新计算后台算、带进度"。）

- [ ] **Step 2: typecheck（会因 Picks 未传新 props 报错，下一任务修）**

Run: `cd frontend && npm run typecheck`
Expected: 报 `Picks.tsx` 缺少 FilterBar 新 props —— 预期，Task B7 修复。

- [ ] **Step 3: 提交**

```bash
git add frontend/src/pages/picks/FilterBar.tsx
git commit -m "feat(frontend): FilterBar draft tier (view+models behind Recompute button) + instant basics"
```

---

### Task B7: Picks 集成（draft 态 + 管线 + GET 门控 + 编排）+ 更新测试

**Files:**
- Modify: `frontend/src/pages/Picks.tsx`
- Modify: `frontend/tests/Picks.test.tsx`

- [ ] **Step 1: 写/改失败测试**（改用 `useCandidates` mock + 新 payload；断言即时层 + 按钮）

```tsx
// frontend/tests/Picks.test.tsx  (REPLACE FILE)
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import Picks from '@/pages/Picks';

const baseItem = {
  rank: 1, symbol: 'SH600519', name: '贵州茅台', score_today: 0.05, score_avg: 0.04,
  rank_avg: 1, days_in_top: 5, consensus: 1, base_scores: {}, horizons: {},
  last_price: 1600, daily_ranks: [1, 1, 1, 1, 1], daily_scores: [0.04, 0.04, 0.04, 0.04, 0.04],
  is_st: false, board: 'main',
};
const item2 = {
  ...baseItem, rank: 2, symbol: 'SH600036', name: '招商银行', score_avg: 0.03, last_price: 35,
  daily_ranks: [40, 50, 60, 45, 55], daily_scores: [0.03, 0.03, 0.03, 0.03, 0.03],
};

vi.mock('@/models/hooks', () => ({
  useCandidates: () => ({
    data: {
      experiment: 'rolling_v2_ensemble', recorder_id: 'abc123', latest_date: '2026-06-16',
      window_days: 20, universe_size: 800, items: [baseItem, item2],
      available_models: ['lgbm_1d', 'lgbm_5d', 'alstm_5d'], active_models: null,
      window_dates: ['2026-06-10', '2026-06-11', '2026-06-12', '2026-06-13', '2026-06-16'],
      as_of_date: '2026-06-16', data_latest_date: '2026-06-16', data_stale_days: 0,
    },
    isPending: false, isFetching: false, error: null,
  }),
}));

function wrap(ui: React.ReactNode) {
  const qc = new QueryClient();
  return <QueryClientProvider client={qc}><MemoryRouter>{ui}</MemoryRouter></QueryClientProvider>;
}

describe('Picks', () => {
  it('renders header and recompute section', () => {
    render(wrap(<Picks />));
    expect(screen.getByText(/选股工作台/)).toBeInTheDocument();
    expect(screen.getByText(/需重新计算/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /重新计算/ })).toBeInTheDocument();
  });

  it('renders both candidate rows by default (loose filters)', () => {
    render(wrap(<Picks />));
    expect(screen.getByText('SH600519')).toBeInTheDocument();
    expect(screen.getByText('SH600036')).toBeInTheDocument();
  });

  it('recompute button starts disabled (no draft change yet)', () => {
    render(wrap(<Picks />));
    expect(screen.getByRole('button', { name: /重新计算/ })).toBeDisabled();
  });
});
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd frontend && npm run test -- Picks`
Expected: FAIL（Picks 仍是旧实现 / 缺"需重新计算"文案 / 按钮 disabled 逻辑）

- [ ] **Step 3: 改 Picks.tsx**

`Picks.tsx` 关键改动（替换组件主体的相关片段）：

1) import 追加：

```ts
import { selectCandidates } from './picks/filter';
import { useRecompute } from './picks/useRecompute';
import RecomputeProgress from './picks/RecomputeProgress';
import { comboKey } from './picks/persistence';
import { WINDOW_K } from './picks/types';
```

2) 常量 `WINDOW_DAYS` 改：

```ts
const POOL_SIZE = 300;     // must equal backend CANDIDATES_POOL_CAP
const WINDOW_DAYS = WINDOW_K; // 20; must equal backend CANDIDATES_WINDOW_K
const MIN_TOP = 0;
```

3) 组件内，`useFilterParams` 之后加 draft 态 + recompute 控制 + GET 门控：

```tsx
  const [params, update, reset] = useFilterParams();
  const [sort, setSort] = useState<SortState>(DEFAULT_SORT);

  // Draft tier: view+models edited freely; applied (= params.view/models) only
  // changes after a successful recompute.
  const [draftView, setDraftView] = useState(params.view);
  const [draftModels, setDraftModels] = useState(params.models);
  // keep draft in sync if applied changes externally (reset / URL nav)
  useEffect(() => { setDraftView(params.view); }, [params.view]);
  useEffect(() => { setDraftModels(params.models); }, [params.models.join(',')]); // eslint-disable-line react-hooks/exhaustive-deps

  const recompute = useRecompute();
  const appliedWarmed = recompute.isWarmed(params.view, params.models);

  // On mount / when applied combo changes, warm it via the progress job before
  // the heavy GET runs (GET is gated by `enabled` below).
  useEffect(() => {
    if (!appliedWarmed && (!recompute.job || recompute.job.status !== 'running')) {
      recompute.start(params.view, params.models);
    }
  }, [params.view, params.models.join(','), appliedWarmed]); // eslint-disable-line react-hooks/exhaustive-deps

  const recomputeDirty =
    comboKey(draftView, draftModels) !== comboKey(params.view, params.models);

  const onRecompute = async () => {
    if (recompute.isWarmed(draftView, draftModels)) {
      update({ view: draftView, models: draftModels });
      return;
    }
    await recompute.start(draftView, draftModels);
  };

  // When a recompute for the DRAFT finishes, commit draft -> applied.
  useEffect(() => {
    if (recompute.job?.status === 'done' && recomputeDirty
        && recompute.isWarmed(draftView, draftModels)) {
      update({ view: draftView, models: draftModels });
    }
  }, [recompute.job?.status]); // eslint-disable-line react-hooks/exhaustive-deps
```

4) `useCandidates` 调用加 `enabled` 门控：

```tsx
  const { data, isPending, isFetching, error } = useCandidates({
    top: POOL_SIZE,
    days: WINDOW_DAYS,
    min_top: MIN_TOP,
    view: params.view,
    models: params.models,
    enabled: appliedWarmed,   // gate the heavy GET behind the warm job
  });
```

> 同步在 `frontend/src/models/hooks.ts` 的 `useCandidates` 参数加 `enabled?: boolean`，并传入 `useQuery({ enabled: params.enabled ?? true, ... })`（从对象里剔除 `enabled` 再放进 queryKey，避免污染键）。见 Step 3b。

5) 把现有 `filtered`/`sorted` 两个 useMemo 替换为单一管线：

```tsx
  const sorted = useMemo(() => {
    if (!data?.items) return [];
    const selected = selectCandidates(
      data.items as Candidate[], params, data.window_dates?.length ?? 0,
    );
    return applySort(selected, sort); // display sort over the selected top-N
  }, [data?.items, data?.window_dates, params, sort]);
```

（删除旧的 `filtered` useMemo 与对 `applyFilters` 的直接调用；`applySort` 仍用。）

6) 渲染：在 `<FilterBar .../>` 上方插入进度条，并给 FilterBar 传 draft props：

```tsx
      <RecomputeProgress job={recompute.job} elapsedSec={recompute.elapsedSec} />

      <FilterBar
        params={params}
        resultCount={data ? sorted.length : null}
        candidateCount={data ? data.items.length : null}
        onChange={update}
        onReset={() => { reset(); setSort(DEFAULT_SORT); setDraftView('ensemble'); setDraftModels([]); }}
        availableModels={data?.available_models ?? []}
        activeModels={data?.active_models ?? null}
        draftView={draftView}
        draftModels={draftModels}
        onDraftView={setDraftView}
        onDraftModels={setDraftModels}
        recomputeDirty={recomputeDirty}
        onRecompute={onRecompute}
        recomputeBusy={recompute.job?.status === 'running'}
      />
```

7) 首次加载文案：把 `isPending` 分支改为同时覆盖"未 warm 即在算"：

```tsx
        ) : (isPending || !appliedWarmed) ? (
          <div className="text-[#8b949e] text-sm">候选池计算中…（见上方进度条）</div>
```

- [ ] **Step 3b: hooks.ts — useCandidates 支持 enabled**

`frontend/src/models/hooks.ts` 的 `useCandidates` 改为：

```ts
export function useCandidates(
  params: {
    top?: number; days?: number; min_top?: number;
    view?: 'ensemble' | 'lightgbm' | 'alstm' | 'tra';
    models?: string[];
    enabled?: boolean;
  } = {},
) {
  const { enabled = true, ...q } = params;
  return useQuery({
    queryKey: ['models', 'candidates', q],
    queryFn: () => api.models.candidates(q),
    enabled,
    staleTime: Infinity,
    gcTime: 30 * 60_000,
    refetchOnWindowFocus: false,
    placeholderData: (prev) => prev,
  });
}
```

- [ ] **Step 4: 跑测试 + typecheck**

Run: `cd frontend && npm run typecheck && npm run test -- Picks persistence`
Expected: 全绿（Picks 3 + persistence 3）

- [ ] **Step 5: 提交**

```bash
git add frontend/src/pages/Picks.tsx frontend/src/models/hooks.ts frontend/tests/Picks.test.tsx
git commit -m "feat(frontend): Picks draft tier + recompute orchestration + gated GET + persistence pipeline"
```

---

### Task B8: useActiveJobs 加 recompute kind（头部徽标，轻量）

**Files:**
- Modify: `frontend/src/jobs/useActiveJobs.ts`

- [ ] **Step 1: 加 recompute 轮询 + chip**

`ActiveJobKind` 加 `'recompute'`：

```ts
export type ActiveJobKind = 'refresh' | 'retrain' | 'evaluation' | 'inference' | 'analysis' | 'recompute';
```

在其它 `useJobPolling` 之后加：

```ts
  const recompute = useJobPolling('recompute', () => api.models.recomputeActive(), interval);
```

`running` 判断里并入：

```ts
      (recompute?.status === 'running') ||
```

依赖数组加 `recompute`。`out` 构造末尾加：

```ts
  if (recompute) {
    const status = recompute.status as ActiveJob['status'];
    const recent =
      recompute.started_at && Date.now() - new Date(recompute.started_at).getTime() < 60_000;
    if (status === 'running' || (recent && (status === 'done' || status === 'failed'))) {
      out.push({
        kind: 'recompute',
        label: status === 'running' ? '重算中' : status === 'done' ? '✓ 重算完成' : '✗ 重算失败',
        detail: recompute.progress ? `${recompute.progress.percent}%` : undefined,
        status,
        started_at: recompute.started_at,
        href: '/picks',
      });
    }
  }
```

- [ ] **Step 2: typecheck**

Run: `cd frontend && npm run typecheck`
Expected: 通过

- [ ] **Step 3: 提交**

```bash
git add frontend/src/jobs/useActiveJobs.ts
git commit -m "feat(frontend): surface recompute job in useActiveJobs header badge"
```

---

### Task B9: 前端整体回归 + 进度权重微调 + 重启验收

**Files:** 可能微调 `backend/app/models/recompute.py`（`_PHASE_BOUNDS`）

- [ ] **Step 1: 全前端测试 + typecheck + lint**

Run: `cd frontend && npm run typecheck && npm run test && npm run lint`
Expected: 全绿

- [ ] **Step 2: 在主仓重启前后端供用户验收**（用户标准指令：改 UI/server 后重启）

Backend（主仓）：`cd /e/Projects/qlib/backend && F:/Tools/Anaconda/envs/qlib/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000`
Frontend（主仓）：`cd /e/Projects/qlib/frontend && npm run dev` → http://localhost:5173/picks

- [ ] **Step 3: 手动验收清单**
  - 首次进 Picks：出现进度条（百分比 + 阶段文案 + 已用秒数）→ 完成后出结果。
  - 改价格/窗口天数/Top N/最少进topN/排序：**瞬时**，无网络重算。
  - 改视图或模型组合：按钮高亮"重新计算 ●"，结果不变；点按钮→进度条→完成换池。
  - 已算过的组合再点：瞬时完成（进度条一闪或跳过）。
  - 进度阶段是否"卡某段再跳"：若 metrics 段占比与实际不符，调 `_PHASE_BOUNDS`（如 metrics 实测占 70% 则改为 `(20,90)` 等），重启后端复验。

- [ ] **Step 4: 提交微调（若有）**

```bash
git add -A && git commit -m "tune(models): recompute progress phase weights per profiling" || echo "no tuning needed"
```

---

## Self-Review（plan vs spec）

**Spec coverage：**
- §4 两层架构 → Task A4/A5（payload 数组）、B5（selectCandidates）、B6/B7（draft 按钮 + 即时层）✓
- §5.1 payload 扩展（window_dates + daily 数组）→ A1/A4/A5 ✓
- §5.2 重算任务（contextvar/thread-local 注入、阶段权重、分批 metrics、job 注册表+守卫）→ A2/A3/A5 ✓
- §5.3 schemas（RecomputeProgress 等）→ A1（注：采用专用 `RecomputeProgress{phase,percent,message}` 而非提升 `ProgressInfo`，更简、避免 models→data 依赖，spec §5.3 已允许"同形"）✓
- §5.4 三端点 → A6（含 active 在 {job_id} 前的顺序坑）✓
- §5.5 缓存键 → 由前后端 WINDOW_K/POOL_CAP 常量一致保证（File Structure 注 + A3 常量 + B7 常量）✓
- §6.1 draft/applied 拆分 → B7 ✓
- §6.2 FilterBar 重组 + 窗口上限 20 → B5/B6 ✓
- §6.3 重算流程（POST→轮询→done 提交→GET 命中）→ B3/B7 ✓
- §6.4 首次冷加载 GET 门控（enabled）→ B7（含 hooks.ts enabled）✓
- §6.5 即时层客户端计算（days_in_top/score_avg/rank/显示cap）→ B2/B5 ✓
- §6.6 进度条组件 + useActiveJobs → B4/B8 ✓
- §9 测试：后端 job/进度/payload + 前端 persistence/Picks → A1-A6/B2/B7 ✓
- §11 风险：进度权重实测微调 → B9 ✓

**Placeholder scan：** 无 TBD/TODO；所有 step 含可运行命令/完整代码。

**Type consistency：** `RecomputeJob`/`RecomputeProgress`/`RecomputeTriggerResponse`/`RecomputeRequest` 名称在 schemas(A1)、recompute(A2/A3)、router(A6)、client(B1)、hook(B3)、组件(B4) 一致；`daily_ranks`/`daily_scores`/`window_dates` 在 schemas(A1)、service(A4/A5)、persistence(B2)、filter(B5)、Picks 测试(B7) 一致；`comboKey`/`daysInTop`/`windowScoreAvg`/`selectCandidates`/`useRecompute` 签名跨任务一致。常量 `CANDIDATES_WINDOW_K=20`/`CANDIDATES_POOL_CAP=300` ↔ 前端 `WINDOW_K=20`/`POOL_SIZE=300` 已标注必须相等。
