# 训练工作台 P1 — 实时训练进度 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a running model retrain show live, structured progress (which stage / which model, X/N, a status line, and a log tail) on a new 训练工作台 page — killing the "看不到进度,像卡死" pain.

**Architecture:** Mirror the *proven* data-refresh progress mechanism. The training subprocess (`production.rolling_train run-once`) prints `PROGRESS {json}` lines to stdout; the backend captures stdout to a per-job log file and tails the latest `PROGRESS` line into the job status; the frontend polls and renders a progress bar (same UI pattern as the data-refresh job on the Dashboard). Training jobs reuse the existing `SchedulerManager` so the concurrency lock + trading-hours guard are shared with cron/run-now (no double-run). A thin new `backend/app/training/` slice + a new `/training` page front this.

**Tech Stack:** Python 3 / FastAPI / APScheduler (backend), React + Vite + TypeScript + TanStack Query + Tailwind (frontend), pytest + vitest.

**Refinement vs spec:** The design spec proposed a separate JSONL progress file gated by `QLIB_PROGRESS_FILE`. During investigation we found the data-refresh job already implements structured progress via `PROGRESS {json}` stdout lines parsed by `backend/app/data/service.py::_latest_progress` (+ `ProgressInfo` schema + `test_progress.py`). P1 mirrors that exact mechanism instead — simpler, consistent, already proven on Windows. Per-epoch loss curves (ALSTM/TRA) need qlib-internal instrumentation and are deferred to P2; P1 delivers robust stage + per-model granularity.

**Out of scope for P1 (later phases):** single-algorithm `--only` re-blend (P3), history table + model comparison + promote (P2), per-epoch loss curves (P2), base-model enable/disable registry UI (P3). Do NOT build these here.

---

## File Structure

**Production (training process side):**
- Create `production/progress.py` — `emit_progress(phase, current, total, message)` prints one `PROGRESS {json}` line to stdout.
- Modify `production/rolling_train.py` — add pure `progress_total(cfg)` helper + `emit_progress(...)` calls at stage boundaries inside `run_once`.

**Backend (`backend/app/`):**
- Modify `app/scheduling/service.py` — job callable gains `(job_id, log_path)`; subprocess stdout is written to the per-job log file; job entries carry `log_path`; raise on non-zero exit so status flips to `failed`.
- Modify `app/main.py` — pass `logs_dir` to `SchedulerManager`.
- Create `app/training/__init__.py`
- Create `app/training/schemas.py` — `TrainingProgress`, `TrainRequest`, `TrainingJobStatus`.
- Create `app/training/service.py` — `latest_progress(log_path)`, `tail_log(log_path)`, `build_job_status(entry)`.
- Create `app/training/router.py` — `POST /api/training/run`, `GET /api/training/jobs/{job_id}`, `GET /api/training/jobs/active`.
- Modify `app/main.py` — register the training router.

**Frontend (`frontend/src/`):**
- Modify `api/client.ts` — add `training` group.
- Create `training/hooks.ts` — `useStartTraining`, `useActiveTrainingJob`, `useTrainingJobDetail`.
- Create `pages/Training.tsx` — 训练 section (full-train trigger) + 进行中 section (progress bar + phase + counts + log tail).
- Modify `App.tsx` — add `/training` route.
- Modify `components/Layout.tsx` — add nav link.

**Tests:**
- `production/tests/test_progress_emit.py`
- `production/tests/test_progress_total.py`
- `backend/app/training/tests/test_progress_parse.py`
- `backend/app/training/tests/test_training_endpoints.py`
- `frontend/src/training/Training.test.tsx` (optional render test)

---

## Task 1: `production/progress.py` — structured progress emitter

**Files:**
- Create: `production/progress.py`
- Test: `production/tests/test_progress_emit.py`

- [ ] **Step 1: Write the failing test**

```python
# production/tests/test_progress_emit.py
import json

from production.progress import emit_progress


def test_emit_progress_prints_parseable_progress_line(capsys):
    emit_progress("train", 3, 9, "training lgbm")
    out = capsys.readouterr().out.strip()
    assert out.startswith("PROGRESS ")
    payload = json.loads(out[len("PROGRESS "):])
    assert payload == {"phase": "train", "current": 3, "total": 9, "message": "training lgbm"}


def test_emit_progress_defaults_empty_message(capsys):
    emit_progress("done", 9, 9)
    out = capsys.readouterr().out.strip()
    payload = json.loads(out[len("PROGRESS "):])
    assert payload["message"] == ""
    assert payload["phase"] == "done"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest production/tests/test_progress_emit.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'production.progress'`

- [ ] **Step 3: Write minimal implementation**

```python
# production/progress.py
"""Structured progress emitter for production training runs.

Prints exactly one line per call:

    PROGRESS {"phase":"train","current":3,"total":9,"message":"training lgbm"}

The backend captures the training subprocess's stdout into a per-job log file
and tails the latest PROGRESS line into the job status. This mirrors the
data-refresh progress mechanism (backend/app/data/service.py::_latest_progress
and production/incremental_refresh.py).

Printing is unconditional and has no side effects beyond stdout, so running a
training script directly from the CLI just shows these lines in the console.
"""
from __future__ import annotations

import json


def emit_progress(phase: str, current: int, total: int, message: str = "") -> None:
    """Emit one structured PROGRESS line to stdout (flushed)."""
    payload = {
        "phase": str(phase),
        "current": int(current),
        "total": int(total),
        "message": str(message),
    }
    # flush=True: the backend tails this promptly even though a retrain runs for
    # many minutes and Python would otherwise buffer stdout when not a tty.
    print("PROGRESS " + json.dumps(payload, ensure_ascii=False), flush=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest production/tests/test_progress_emit.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add production/progress.py production/tests/test_progress_emit.py
git commit -m "feat(training): production.progress.emit_progress — structured PROGRESS stdout line"
```

---

## Task 2: Instrument `rolling_train.run_once` with stage progress

**Files:**
- Modify: `production/rolling_train.py` (add `progress_total` near line 64; add `emit_progress` calls in `run_once`, lines 405-565)
- Test: `production/tests/test_progress_total.py`

**Context:** `run_once` (line 337) trains each enabled base model for all horizons (loop at lines 414-426), then runs the ensemble step (line 436+), writes `pred.pkl` (line 486), and returns (line 565). We emit one unit after universe build, one before each enabled model, one before the ensemble step, and one when done. `progress_total` must agree with that stepping.

- [ ] **Step 1: Write the failing test for the pure helper**

```python
# production/tests/test_progress_total.py
from production.rolling_train import progress_total
from production.walk_forward import HorizonConfig


def _cfg(model_ids_enabled):
    # Minimal stand-in: progress_total only reads .model_specs and .horizons.
    class _C:
        model_specs = [{"id": m, "enabled": en} for m, en in model_ids_enabled]
        horizons = [
            HorizonConfig(name="1d", train_years=3, valid_years=1, stack_years=1, test_weeks=1),
            HorizonConfig(name="5d", train_years=5, valid_years=1, stack_years=1, test_weeks=1),
            HorizonConfig(name="20d", train_years=7, valid_years=1, stack_years=1, test_weeks=1),
        ]
    return _C()


def test_progress_total_counts_enabled_models_plus_fixed_stages():
    # 3 enabled models + universe(1) + ensemble(1) + done(1) = 6
    assert progress_total(_cfg([("lgbm", True), ("alstm", True), ("tra", True)])) == 6


def test_progress_total_ignores_disabled_models():
    # 1 enabled + 3 fixed = 4
    assert progress_total(_cfg([("lgbm", True), ("alstm", False), ("tra", False)])) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest production/tests/test_progress_total.py -v`
Expected: FAIL — `ImportError: cannot import name 'progress_total'`

- [ ] **Step 3: Add the pure helper**

Add directly below `DEFAULT_MODELS = ("lgbm", "alstm", "tra")` (line 64) in `production/rolling_train.py`:

```python
def progress_total(cfg: "RollingConfig") -> int:
    """Total progress units for one run_once: one per enabled base model, plus
    three fixed stages (universe build, ensemble, done). Must stay in sync with
    the emit_progress calls in run_once."""
    enabled = [s for s in cfg.model_specs if s.get("enabled")]
    return len(enabled) + 3
```

- [ ] **Step 4: Run test to verify the helper passes**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest production/tests/test_progress_total.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Wire emit_progress into run_once**

In `production/rolling_train.py`, add the import near the other `from production...` imports (top of file, ~line 54):

```python
from production.progress import emit_progress
```

Immediately after `build_universe` logs `universe_built` (after line 410, before the `series_list: list[pd.Series] = []` line 413), insert:

```python
    total = progress_total(cfg)
    step = 1
    emit_progress("universe", step, total, f"universe {len(members)} stocks")
```

Replace the model loop (lines 414-426) so each enabled model emits before training:

```python
    # Train all enabled base models for all horizons
    series_list: list[pd.Series] = []
    for spec in cfg.model_specs:
        if not spec["enabled"]:
            continue
        step += 1
        emit_progress("train", step, total, f"training {spec['id']}")
        if spec["id"] == "lgbm":
            for h in cfg.horizons:
                s = train_lgbm_horizon(cfg, h, universe_name, end_date, features=features, objective=objective)
                series_list.append(s)
        elif spec["id"] == "alstm":
            from production.train_alstm import train_alstm_multihead  # added in T13
            series_list.extend(train_alstm_multihead(cfg, universe_name, end_date))
        elif spec["id"] == "tra":
            from production.train_tra import train_tra_multihead  # added in T15
            series_list.extend(train_tra_multihead(cfg, universe_name, end_date))
```

Immediately before the ensemble comment block (`# Ensemble step — Phase E`, ~line 438), after `base_preds = pd.concat(...)` (line 436), insert:

```python
    step += 1
    emit_progress("ensemble", step, total, "stacking ensemble")
```

Immediately before `return pred_path` at the end of `run_once` (line 565), insert:

```python
    emit_progress("done", total, total, "done")
```

- [ ] **Step 6: Run the full production test subset to confirm no regression**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest production/tests/test_progress_total.py production/tests/test_progress_emit.py -v`
Expected: PASS (4 passed). The emits are plain prints — they do not affect `run_once`'s return value or existing rolling-train tests.

- [ ] **Step 7: Commit**

```bash
git add production/rolling_train.py production/tests/test_progress_total.py
git commit -m "feat(training): emit stage progress (universe/per-model/ensemble/done) from run_once"
```

---

## Task 3: SchedulerManager — per-job log file + log_path on entries + fail on non-zero exit

**Files:**
- Modify: `backend/app/scheduling/service.py` (`JobCallable` line 56; `make_subprocess_retrain_job` lines 59-89; `SchedulerManager.__init__` lines 101-121; `run_now` entry lines 240-248; `_gated_job_fn` lines 264-320)
- Modify: `backend/app/main.py` (lines 44-49)
- Test: `backend/app/scheduling/tests/test_job_log_path.py`

**Context:** Today the job callable takes no args and drains stdout only to the logger (lines 81-82). We change it to `(job_id, log_path)`, write stdout bytes to `log_path`, and `raise` on non-zero exit so `_gated_job_fn` marks the job `failed`. The manager computes `log_path` from a `logs_dir` and stores it on each job entry so the training layer can tail progress.

- [ ] **Step 1: Write the failing test**

```python
# backend/app/scheduling/tests/test_job_log_path.py
import asyncio
from pathlib import Path

import pytest

from app.scheduling.service import SchedulerManager


@pytest.mark.asyncio
async def test_run_now_sets_log_path_and_invokes_job_with_it(tmp_path, monkeypatch):
    captured = {}

    async def fake_job(job_id: str, log_path: Path) -> None:
        captured["job_id"] = job_id
        captured["log_path"] = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text('PROGRESS {"phase":"done","current":1,"total":1,"message":"ok"}\n', encoding="utf-8")

    mgr = SchedulerManager(fake_job, logs_dir=tmp_path)
    # Bypass APScheduler: drive _gated_job_fn directly with a pre-registered entry.
    job_id = "retrain_weekly_manual_test"
    mgr._remember_job(job_id, {
        "job_id": job_id, "kind": "manual", "status": "pending",
        "started_at": None, "finished_at": None, "queued_at": "t", "error": None,
        "log_path": str(tmp_path / f"api_retrain_{job_id}.log"),
    })
    mgr._active_job_id = job_id
    await mgr._gated_job_fn(_tracked_job_id=job_id)

    entry = mgr.get_job_status(job_id)
    assert entry["status"] == "done"
    assert entry["log_path"] == str(tmp_path / f"api_retrain_{job_id}.log")
    assert captured["job_id"] == job_id
    assert captured["log_path"] == tmp_path / f"api_retrain_{job_id}.log"


@pytest.mark.asyncio
async def test_job_raising_marks_failed(tmp_path):
    async def boom(job_id: str, log_path: Path) -> None:
        raise RuntimeError("rolling_train exited 1")

    mgr = SchedulerManager(boom, logs_dir=tmp_path)
    job_id = "retrain_weekly_manual_boom"
    mgr._remember_job(job_id, {
        "job_id": job_id, "kind": "manual", "status": "pending",
        "started_at": None, "finished_at": None, "queued_at": "t", "error": None,
        "log_path": str(tmp_path / f"api_retrain_{job_id}.log"),
    })
    with pytest.raises(RuntimeError):
        await mgr._gated_job_fn(_tracked_job_id=job_id)
    assert mgr.get_job_status(job_id)["status"] == "failed"
    assert "exited 1" in mgr.get_job_status(job_id)["error"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend; F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest app/scheduling/tests/test_job_log_path.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'logs_dir'` (and the job-fn signature mismatch).

- [ ] **Step 3: Update `JobCallable`, `make_subprocess_retrain_job`, and `__init__`**

In `backend/app/scheduling/service.py`, change the type alias (line 56):

```python
JobCallable = Callable[[str, Path], Awaitable[None]]
```

Replace `make_subprocess_retrain_job` (lines 59-89) with:

```python
def make_subprocess_retrain_job(python_path: str, repo_root: Path) -> JobCallable:
    """Return an async job that spawns `python -m production.rolling_train run-once`
    as a child process and writes its stdout to a per-job log file. Required
    because rolling_train blocks for many minutes of CPU/GPU work; running it
    inside the FastAPI event loop would freeze HTTP. The log file is tailed by
    the training layer for structured PROGRESS lines.
    """

    async def _job(job_id: str, log_path: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _log.info("retrain_subprocess_starting", job_id=job_id, python=python_path, cwd=str(repo_root))
        proc = await asyncio.create_subprocess_exec(
            python_path,
            "-m",
            "production.rolling_train",
            "run-once",
            cwd=str(repo_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        if proc.stdout is None:
            _log.error("retrain_subprocess_no_stdout", job_id=job_id)
            await proc.wait()
            raise RuntimeError("retrain subprocess produced no stdout pipe")
        with log_path.open("wb") as fh:
            async for line in proc.stdout:
                fh.write(line)
                fh.flush()
                _log.info("retrain_subprocess_stdout", line=line.decode(errors="replace").rstrip())
        rc = await proc.wait()
        if rc != 0:
            _log.error("retrain_subprocess_failed", job_id=job_id, returncode=rc)
            raise RuntimeError(f"rolling_train exited {rc}")
        _log.info("retrain_subprocess_ok", job_id=job_id)

    return _job
```

Update `SchedulerManager.__init__` signature (line 101) and add `_logs_dir`:

```python
    def __init__(self, job_fn: JobCallable, logs_dir: Path | None = None):
        self._scheduler = AsyncIOScheduler()
        self._raw_job_fn = job_fn
        self._running_lock = asyncio.Lock()
        self._logs_dir = logs_dir or (Path(__file__).resolve().parent.parent.parent.parent / "logs")
        self._started = False
        self._pending_count = 0
        from collections import OrderedDict as _OD
        self._MAX_JOBS = 50
        self._jobs: "_OD[str, dict]" = _OD()
        self._active_job_id: str | None = None

    def _log_path_for(self, job_id: str) -> Path:
        return self._logs_dir / f"api_retrain_{job_id}.log"
```

- [ ] **Step 4: Add `log_path` to job entries + pass it to the job fn**

In `run_now` (lines 240-248), add `log_path` to the pre-registered entry dict:

```python
            self._remember_job(job_id, {
                "job_id": job_id,
                "kind": "manual",
                "status": "pending",
                "started_at": None,
                "finished_at": None,
                "queued_at": datetime.now(tz=_CST).isoformat(),
                "error": None,
                "log_path": str(self._log_path_for(job_id)),
            })
```

In `_gated_job_fn`, in the cron-mint branch (lines 282-290), add `log_path` likewise:

```python
            self._remember_job(_tracked_job_id, {
                "job_id": _tracked_job_id,
                "kind": "cron",
                "status": "pending",
                "started_at": None,
                "finished_at": None,
                "queued_at": datetime.now(tz=_CST).isoformat(),
                "error": None,
                "log_path": str(self._log_path_for(_tracked_job_id)),
            })
```

Change the job invocation (line 306) from `await self._raw_job_fn()` to:

```python
                    await self._raw_job_fn(_tracked_job_id, self._log_path_for(_tracked_job_id))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend; F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest app/scheduling/tests/test_job_log_path.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Update the lifespan wiring + run the scheduling regression**

In `backend/app/main.py`, pass `logs_dir` (lines 44-49):

```python
    repo_root = Path(__file__).resolve().parent.parent.parent
    retrain_job = make_subprocess_retrain_job(
        python_path=settings.retrain_python_path,
        repo_root=repo_root,
    )

    manager = SchedulerManager(retrain_job, logs_dir=repo_root / "logs")
```

Run the full scheduling test suite to confirm no regression:

Run: `cd backend; F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest app/scheduling/ -v`
Expected: PASS (all existing scheduling tests + the 2 new ones).

- [ ] **Step 7: Commit**

```bash
git add backend/app/scheduling/service.py backend/app/main.py backend/app/scheduling/tests/test_job_log_path.py
git commit -m "feat(training): SchedulerManager writes per-job log + carries log_path + fails on non-zero exit"
```

---

## Task 4: `backend/app/training/` slice — schemas + progress parsing + job-status builder

**Files:**
- Create: `backend/app/training/__init__.py` (empty)
- Create: `backend/app/training/schemas.py`
- Create: `backend/app/training/service.py`
- Test: `backend/app/training/tests/__init__.py` (empty), `backend/app/training/tests/test_progress_parse.py`

**Context:** Mirror `backend/app/data/service.py::_latest_progress` (line 441) and `_tail_log` (line 419) and `ProgressInfo` (data/schemas.py line 4). We keep the training slice self-contained (its own `TrainingProgress`) rather than importing from the data slice, matching the vertical-slice style. `build_job_status` enriches a SchedulerManager job entry (which now carries `log_path`) with parsed progress + log tail.

- [ ] **Step 1: Write the failing test**

```python
# backend/app/training/tests/test_progress_parse.py
from pathlib import Path

from app.training.service import latest_progress, tail_log, build_job_status


def test_latest_progress_parses_last_progress_line(tmp_path: Path):
    log = tmp_path / "j.log"
    log.write_text(
        'PROGRESS {"phase":"universe","current":1,"total":6,"message":"u"}\n'
        'some noise\n'
        'PROGRESS {"phase":"train","current":2,"total":6,"message":"training lgbm"}\n',
        encoding="utf-8",
    )
    p = latest_progress(log)
    assert p is not None
    assert p.phase == "train"
    assert p.current == 2
    assert p.total == 6
    assert p.message == "training lgbm"


def test_latest_progress_missing_or_empty_returns_none(tmp_path: Path):
    assert latest_progress(tmp_path / "absent.log") is None
    empty = tmp_path / "e.log"
    empty.write_text("plain output, no progress\n", encoding="utf-8")
    assert latest_progress(empty) is None


def test_build_job_status_enriches_entry_with_progress(tmp_path: Path):
    log = tmp_path / "j.log"
    log.write_text('PROGRESS {"phase":"done","current":6,"total":6,"message":"done"}\n', encoding="utf-8")
    entry = {
        "job_id": "j1", "kind": "manual", "status": "done",
        "started_at": "s", "finished_at": "f", "error": None, "log_path": str(log),
    }
    st = build_job_status(entry)
    assert st.job_id == "j1"
    assert st.status == "done"
    assert st.progress is not None and st.progress.phase == "done"
    assert "PROGRESS" in (st.log_tail or "")


def test_build_job_status_handles_missing_log_path(tmp_path: Path):
    entry = {
        "job_id": "j2", "kind": "manual", "status": "pending",
        "started_at": None, "finished_at": None, "error": None, "log_path": None,
    }
    st = build_job_status(entry)
    assert st.progress is None
    assert st.log_tail is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend; F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest app/training/tests/test_progress_parse.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.training'`

- [ ] **Step 3: Create the package + schemas**

```python
# backend/app/training/__init__.py
```

```python
# backend/app/training/schemas.py
from pydantic import BaseModel, Field


class TrainingProgress(BaseModel):
    phase: str = Field(..., description='"universe" | "train" | "ensemble" | "done" (or future phases)')
    current: int = Field(..., description="Current step (1-based); equals total when finished")
    total: int = Field(..., description="Total steps for this run")
    message: str = Field("", description="Human-readable status line, e.g. 'training lgbm'")


class TrainRequest(BaseModel):
    # P1: full ensemble only. `scope`/`models` reserved for P3 (single-algo).
    scope: str = Field("full", description='"full" (P1). Single-algo arrives in P3.')
    force: bool = Field(False, description="Override the trading-hours guard.")


class TrainingJobStatus(BaseModel):
    job_id: str
    kind: str = Field(..., description='"cron" | "manual"')
    status: str = Field(..., description='"pending" | "running" | "done" | "failed" | "skipped"')
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    progress: TrainingProgress | None = None
    log_tail: str | None = None
```

- [ ] **Step 4: Create the service (progress parsing + status builder)**

```python
# backend/app/training/service.py
"""Training job helpers: parse structured PROGRESS lines from a job's log file
and assemble a TrainingJobStatus from a SchedulerManager job entry.

Mirrors backend/app/data/service.py::_latest_progress / _tail_log (the proven
data-refresh progress mechanism). Kept self-contained to the training slice.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.training.schemas import TrainingJobStatus, TrainingProgress


def tail_log(log_path: Path, n_lines: int = 50) -> str | None:
    if not log_path.is_file():
        return None
    try:
        with log_path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            read_size = min(size, 64 * 1024)
            f.seek(size - read_size)
            data = f.read()
        text = data.decode("utf-8", errors="replace")
        return "\n".join(text.splitlines()[-n_lines:])
    except Exception:
        return None


def latest_progress(log_path: Path) -> TrainingProgress | None:
    """Return the latest parseable 'PROGRESS {json}' line, or None."""
    if not log_path.is_file():
        return None
    try:
        with log_path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            read_size = min(size, 32 * 1024)
            f.seek(size - read_size)
            data = f.read()
        text = data.decode("utf-8", errors="replace")
        for line in reversed(text.splitlines()):
            line = line.strip()
            if not line.startswith("PROGRESS "):
                continue
            try:
                payload = json.loads(line[len("PROGRESS "):])
                return TrainingProgress(**payload)
            except Exception:
                continue
        return None
    except Exception:
        return None


def build_job_status(entry: dict) -> TrainingJobStatus:
    """Enrich a SchedulerManager job entry with parsed progress + log tail."""
    log_path_str = entry.get("log_path")
    progress = None
    log_tail = None
    if log_path_str:
        lp = Path(log_path_str)
        progress = latest_progress(lp)
        log_tail = tail_log(lp)
    return TrainingJobStatus(
        job_id=entry["job_id"],
        kind=entry.get("kind", "manual"),
        status=entry["status"],
        started_at=entry.get("started_at"),
        finished_at=entry.get("finished_at"),
        error=entry.get("error"),
        progress=progress,
        log_tail=log_tail,
    )
```

```python
# backend/app/training/tests/__init__.py
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend; F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest app/training/tests/test_progress_parse.py -v`
Expected: PASS (4 passed)

- [ ] **Step 6: Commit**

```bash
git add backend/app/training/__init__.py backend/app/training/schemas.py backend/app/training/service.py backend/app/training/tests/
git commit -m "feat(training): training slice schemas + PROGRESS parsing + job-status builder"
```

---

## Task 5: `backend/app/training/router.py` — endpoints + register in main

**Files:**
- Create: `backend/app/training/router.py`
- Modify: `backend/app/main.py` (imports near line 23; `include_router` near line 91)
- Test: `backend/app/training/tests/test_training_endpoints.py`

**Context:** The training router reuses the shared `SchedulerManager` singleton via `app.scheduling.router.get_manager()` — so the concurrency lock + trading-hours guard are shared with cron and the settings page run-now (no double-run). P1 `POST /api/training/run` triggers a full retrain (delegates to `run_now`).

- [ ] **Step 1: Write the failing test**

```python
# backend/app/training/tests/test_training_endpoints.py
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app
from app.scheduling.router import set_manager


class _FakeManager:
    def __init__(self):
        self._entry = None

    async def run_now(self, session, force=False):
        self._entry = {
            "job_id": "tjob1", "kind": "manual", "status": "running",
            "started_at": "s", "finished_at": None, "error": None, "log_path": None,
        }
        return "tjob1"

    def get_job_status(self, job_id):
        if self._entry and self._entry["job_id"] == job_id:
            return self._entry
        return None

    def get_active_job(self):
        return self._entry


@pytest.mark.asyncio
async def test_run_then_status_and_active(monkeypatch):
    app = create_app()
    fake = _FakeManager()
    set_manager(fake)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post("/api/training/run", json={"scope": "full", "force": True})
        assert r.status_code == 200
        job_id = r.json()["job_id"]
        assert job_id == "tjob1"

        r2 = await ac.get(f"/api/training/jobs/{job_id}")
        assert r2.status_code == 200
        assert r2.json()["status"] == "running"

        r3 = await ac.get("/api/training/jobs/active")
        assert r3.status_code == 200
        assert r3.json()["job_id"] == "tjob1"


@pytest.mark.asyncio
async def test_status_unknown_job_returns_null():
    app = create_app()
    set_manager(_FakeManager())
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/api/training/jobs/nope")
        assert r.status_code == 200
        assert r.json() is None
```

> Note: `create_app()` does not run the lifespan, so `set_manager(fake)` installs the manager the routes read. The DB session dependency is unused by these routes (run_now is faked), so no DB setup is needed.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend; F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest app/training/tests/test_training_endpoints.py -v`
Expected: FAIL — 404s (router not registered) / `ModuleNotFoundError: app.training.router`.

- [ ] **Step 3: Create the router**

```python
# backend/app/training/router.py
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_session
from app.scheduling.router import get_manager
from app.scheduling.service import AlreadyRunning, TradingHoursViolation
from app.training.schemas import TrainingJobStatus, TrainRequest

router = APIRouter()


@router.post("/run")
async def run_training(
    payload: TrainRequest,
    session: AsyncSession = Depends(get_session),
):
    """Start a full retrain (P1). Reuses the shared SchedulerManager so the
    concurrency lock + trading-hours guard are shared with cron/run-now."""
    try:
        job_id = await get_manager().run_now(session, force=payload.force)
        return {"status": "started", "job_id": job_id}
    except TradingHoursViolation as exc:
        return {"status": "rejected", "reason": str(exc)}
    except AlreadyRunning as exc:
        return {"status": "rejected", "reason": str(exc)}


@router.get("/jobs/active", response_model=TrainingJobStatus | None)
def active_job():
    from app.training.service import build_job_status
    entry = get_manager().get_active_job()
    return build_job_status(entry) if entry is not None else None


@router.get("/jobs/{job_id}", response_model=TrainingJobStatus | None)
def job_status(job_id: str):
    from app.training.service import build_job_status
    entry = get_manager().get_job_status(job_id)
    return build_job_status(entry) if entry is not None else None
```

> Route order: `/jobs/active` is declared before `/jobs/{job_id}` so "active" is never captured as a job_id.

- [ ] **Step 4: Register the router in main.py**

In `backend/app/main.py`, add the import next to the other routers (after line 23):

```python
from app.training.router import router as training_router
```

Add the include next to the others (after the scheduling include, line 91):

```python
    app.include_router(training_router, prefix="/api/training", tags=["training"])
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend; F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest app/training/tests/test_training_endpoints.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add backend/app/training/router.py backend/app/main.py backend/app/training/tests/test_training_endpoints.py
git commit -m "feat(training): /api/training run + job-status endpoints (shared SchedulerManager)"
```

---

## Task 6: Frontend API client — `training` group

**Files:**
- Modify: `frontend/src/api/client.ts` (add `training` to the `api` object, next to `scheduling`)

**Context:** Mirror the existing `scheduling` group and the typed `request<R>(...)` helper. Since `/api/training/*` is hand-written (not yet in the generated OpenAPI types), declare inline response types matching `TrainingJobStatus`.

- [ ] **Step 1: Add the types + client group**

In `frontend/src/api/client.ts`, add near the top-level type declarations:

```tsx
export interface TrainingProgress {
  phase: string;
  current: number;
  total: number;
  message: string;
}

export interface TrainingJobStatus {
  job_id: string;
  kind: 'cron' | 'manual';
  status: 'pending' | 'running' | 'done' | 'failed' | 'skipped';
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
  progress: TrainingProgress | null;
  log_tail: string | null;
}

export interface StartTrainingResponse {
  status: 'started' | 'rejected';
  job_id?: string;
  reason?: string;
}
```

Inside the `api` object, add a `training` group (place it right after the `scheduling: { ... }` block):

```tsx
  training: {
    run: (force = false) =>
      request<StartTrainingResponse>('/api/training/run', {
        method: 'POST',
        body: JSON.stringify({ scope: 'full', force }),
      }),
    active: () => request<TrainingJobStatus | null>('/api/training/jobs/active'),
    status: (jobId: string) =>
      request<TrainingJobStatus | null>(`/api/training/jobs/${encodeURIComponent(jobId)}`),
  },
```

- [ ] **Step 2: Type-check**

Run: `cd frontend; npm run build`
Expected: build succeeds (TypeScript compiles). If `request` requires a content-type header for POST bodies, match the existing `scheduling.putRetrain` call's options exactly.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/api/client.ts
git commit -m "feat(training): frontend api.training client group + types"
```

---

## Task 7: Frontend hooks — start + poll training job

**Files:**
- Create: `frontend/src/training/hooks.ts`

**Context:** Mirror `frontend/src/scheduling/hooks.ts` (useQuery/useMutation) and `frontend/src/jobs/useJobPolling.ts` (poll at 3s while running). The detail hook polls only while the job is `running`/`pending`.

- [ ] **Step 1: Create the hooks**

```tsx
// frontend/src/training/hooks.ts
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { api } from '@/api/client';

export function useStartTraining() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (force: boolean) => api.training.run(force),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['training', 'active'] }),
  });
}

export function useActiveTrainingJob() {
  return useQuery({
    queryKey: ['training', 'active'],
    queryFn: () => api.training.active(),
    refetchInterval: (q) => {
      const s = q.state.data?.status;
      return s === 'running' || s === 'pending' ? 3_000 : false;
    },
    refetchIntervalInBackground: true,
  });
}

export function useTrainingJobDetail(jobId: string | null) {
  return useQuery({
    queryKey: ['training', 'job', jobId],
    queryFn: () => api.training.status(jobId as string),
    enabled: !!jobId,
    refetchInterval: (q) => (q.state.data?.status === 'running' ? 3_000 : false),
    refetchIntervalInBackground: true,
  });
}
```

> If the installed `@tanstack/react-query` version types `refetchInterval`'s argument as the query (v5) vs the data (v4), match whatever `frontend/src/jobs/useJobPolling.ts` uses. v5 passes the query object (`q.state.data`), which is assumed here.

- [ ] **Step 2: Type-check**

Run: `cd frontend; npm run build`
Expected: build succeeds.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/training/hooks.ts
git commit -m "feat(training): frontend hooks — start training + poll active/detail"
```

---

## Task 8: Frontend page `Training.tsx` + route + nav

**Files:**
- Create: `frontend/src/pages/Training.tsx`
- Modify: `frontend/src/App.tsx` (import + `<Route>`)
- Modify: `frontend/src/components/Layout.tsx` (nav `<NavLink>`)

**Context:** P1 page has two sections: 训练 (a "立即训练(全量)" button → `useStartTraining`) and 进行中 (live progress bar + phase + X/N + message + log tail, driven by `useActiveTrainingJob`). Reuse the Dashboard progress-bar markup (Dashboard.tsx lines 202-234) and the dark Tailwind palette.

- [ ] **Step 1: Create the page**

```tsx
// frontend/src/pages/Training.tsx
import { useActiveTrainingJob, useStartTraining } from '@/training/hooks';

const PHASE_LABEL: Record<string, string> = {
  universe: '构建股票池',
  train: '训练模型',
  ensemble: '融合',
  done: '完成',
};

function PhaseBadge({ phase }: { phase: string }) {
  return (
    <span className="inline-block px-2 py-0.5 rounded text-xs bg-[#1f6feb] text-white font-medium whitespace-nowrap">
      {PHASE_LABEL[phase] ?? phase}
    </span>
  );
}

export default function Training() {
  const start = useStartTraining();
  const { data: job } = useActiveTrainingJob();
  const running = job?.status === 'running' || job?.status === 'pending';

  return (
    <div className="p-4 space-y-6 text-[#e6edf3]">
      <h1 className="text-lg font-semibold">训练工作台</h1>

      {/* 训练 section */}
      <section className="rounded-lg border border-[#30363d] p-4 space-y-3">
        <h2 className="text-sm font-medium text-[#8b949e]">训练</h2>
        <button
          className="px-3 py-1.5 rounded bg-[#1f6feb] text-white text-sm disabled:opacity-50"
          disabled={running || start.isPending}
          onClick={() => start.mutate(false)}
        >
          {running ? '训练进行中…' : '立即训练(全量)'}
        </button>
        {start.data?.status === 'rejected' && (
          <p className="text-xs text-amber-400">已拒绝:{start.data.reason}</p>
        )}
        {start.isError && (
          <p className="text-xs text-red-400">启动失败:{String((start.error as Error)?.message ?? start.error)}</p>
        )}
      </section>

      {/* 进行中 section */}
      <section className="rounded-lg border border-[#30363d] p-4 space-y-3">
        <h2 className="text-sm font-medium text-[#8b949e]">进行中</h2>
        {!job && <p className="text-xs text-[#8b949e]">本进程暂无训练任务。</p>}
        {job && (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-xs">
              <span className="text-[#8b949e]">任务 {job.job_id}</span>
              <span className="font-mono">{job.status}</span>
            </div>
            {job.status === 'running' && job.progress && (
              <div>
                <div className="flex justify-between text-xs text-[#8b949e] mb-1 gap-2">
                  <span className="flex items-center min-w-0">
                    <PhaseBadge phase={job.progress.phase} />
                    {job.progress.message && <span className="ml-2 truncate">{job.progress.message}</span>}
                  </span>
                  <span className="font-mono whitespace-nowrap">
                    {job.progress.current}/{job.progress.total}
                  </span>
                </div>
                <div className="w-full h-2 bg-[#21262d] rounded-full overflow-hidden">
                  <div
                    className="h-full bg-[#1f6feb] transition-all"
                    style={{
                      width: `${
                        job.progress.total > 0
                          ? Math.min(100, (job.progress.current / job.progress.total) * 100)
                          : 0
                      }%`,
                    }}
                  />
                </div>
              </div>
            )}
            {job.status === 'running' && !job.progress && (
              <p className="text-xs text-[#8b949e]">初始化中…</p>
            )}
            {job.status === 'failed' && (
              <p className="text-xs text-red-400">训练失败:{job.error}</p>
            )}
            {job.log_tail && (
              <pre className="mt-2 max-h-48 overflow-auto rounded bg-[#0d1117] border border-[#21262d] p-2 text-[11px] leading-relaxed text-[#8b949e] whitespace-pre-wrap">
                {job.log_tail}
              </pre>
            )}
          </div>
        )}
      </section>
    </div>
  );
}
```

- [ ] **Step 2: Add the route**

In `frontend/src/App.tsx`, add the import with the other page imports:

```tsx
import Training from '@/pages/Training';
```

Add the route inside `<Route element={<Layout />}>` (next to `/settings`):

```tsx
        <Route path="/training" element={<Training />} />
```

- [ ] **Step 3: Add the nav link**

In `frontend/src/components/Layout.tsx`, add to the `<nav>` block:

```tsx
          <NavLink to="/training">训练工作台</NavLink>
```

- [ ] **Step 4: Type-check + build**

Run: `cd frontend; npm run build`
Expected: build succeeds.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/Training.tsx frontend/src/App.tsx frontend/src/components/Layout.tsx
git commit -m "feat(training): 训练工作台 page (full-train trigger + live progress) + route + nav"
```

---

## Task 9: Backend regression + live smoke

**Files:** none (verification only)

- [ ] **Step 1: Full backend test suite**

Run: `cd backend; F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest -q`
Expected: all pass (existing + new training/scheduling tests). Investigate any failure before proceeding.

- [ ] **Step 2: Live smoke — observe real progress**

Start the backend (`start.bat` or `cd backend; uvicorn app.main:app --host 127.0.0.1 --port 8000`), open the app, go to 训练工作台, click 立即训练(全量) (outside trading hours, or it will be rejected — pass force via the button if needed during testing). Within ~30s confirm:
- the 进行中 section shows a phase badge (构建股票池 → 训练模型 …) and an advancing X/N bar;
- the log tail shows `PROGRESS {...}` lines;
- on completion the status becomes `done` (or `failed` with an error if the run fails).

If progress never appears, check `logs/api_retrain_<job_id>.log` exists and contains `PROGRESS` lines, and that the subprocess `python_path` (`settings.retrain_python_path`) is the qlib env.

- [ ] **Step 3: Commit any fixes discovered during smoke**

```bash
git add -A
git commit -m "fix(training): address P1 live-smoke findings"
```

---

## Self-Review (completed during authoring)

**Spec coverage (P1 scope only):**
- "实时训练进度 (治本)" → Tasks 1-2 (emit) + 3-4 (capture/parse) + 7-8 (poll/render). ✓
- 进度治本机制 (PROGRESS line + backend tail + polling) → Tasks 1,3,4,7. ✓ (refined from JSONL to the proven `PROGRESS` stdout mechanism — noted in header.)
- 新页面「训练工作台」(P1 = 训练 + 进行中 sections) → Task 8. ✓ (历史/对比/单算法/registry explicitly deferred to P2/P3.)
- 复用并扩展 SchedulerManager (shared lock/guard) → Tasks 3,5. ✓
- 损失曲线 → deferred to P2 (needs qlib pytorch-model instrumentation; header + Task notes state this). ✓

**Placeholder scan:** No TBD/TODO; every code step has complete code. ✓
**Type consistency:** `TrainingProgress`/`TrainingJobStatus` fields identical across schemas.py, service.py, client.ts, hooks.ts, Training.tsx. `emit_progress(phase,current,total,message)` ↔ parser keys ↔ `progress_total` units (`len(enabled)+3`) consistent. Job-fn signature `(job_id, log_path)` consistent in service.py + main.py + tests. ✓
**Scope:** Single shippable increment (full-train progress visible). ✓
