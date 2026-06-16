# 训练工作台 P3a — 单算法候选训练 + Promote Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Let the user retrain ONE base model (LGBM/ALSTM/TRA) on the current production fold, re-blend it with the OTHER models' cached predictions into a CANDIDATE ensemble (does NOT auto-go-live), see it in history, compare it, and PROMOTE it to production when satisfied.

**Architecture (grounded by investigation):**
- **Re-blend is cheap & exact.** Each base model's recorder (`<model>_<h>_<end_date>`) persists full-window `pred_<h>.pkl` (train+valid+test). `run_once`'s ensemble block already refits the Ridge stacker from `base_preds` (masks the valid window + fetches fresh 5d labels) — it does NOT need a persisted stacker. So single-algo re-blend = retrain the target model, LOAD the other enabled models' `pred_<h>.pkl` from their existing recorders (same `end_date`), assemble `series_list`, and run the SAME ensemble logic. No DL retrain (~1–2h saved). Same `end_date` ⇒ deterministic same universe/splits.
- **Candidates don't auto-go-live.** Serving picks current = newest recorder *with a loadable pred.pkl* in the **production experiment** (`get_latest_recorder_id`). Candidates are written to a SEPARATE candidate experiment, so serving ignores them. They still appear in the evaluation recorder scan (history/compare). **Promote** = load the chosen candidate's `pred.pkl` → save it into a NEW recorder in the production experiment (becomes newest → current). This is mlflow-correct (no cross-experiment directory surgery) and reuses `R.start`/`save_objects`.

**Tech Stack:** Python / qlib MLflow recorders / FastAPI / SQLAlchemy (backend); React + TanStack Query (frontend).

**Scope (this slice = P3a only):** single-algo re-blend candidate + promote, end to end. **OUT (later P3 slices):** model-registry enable-toggles writing yaml (P3b), per-epoch loss curves (P3c), `created_at` format unification (P3d). Do NOT build these here.

**Confirmed design choices (from investigation + user "继续"):** re-blend via REFIT on cached OOF (not reuse-old-weights); candidates in a separate experiment; promote = copy pred.pkl into a fresh production recorder.

---

## Key facts (verified — cite in tasks)
- `production/rolling_train.py::run_once` ensemble block (~lines 451–506): builds `base_preds = pd.concat(series_list)`, fetches 5d valid labels via `D.features`, masks `valid_base`, `RidgeStacker().fit_oof(valid_base, labels["y"])`, `predict_with_fallback(base_preds)` → `score`, `consensus_per_row`, `ewma_smooth`, `write_pred_pkl`, then persists `ensemble_<end_date>` recorder.
- Per-model recorder load pattern: `production/run_split.py:198–237` — match `run_name == f"{model_id}_{h}_{end_str}"`, `rec.load_object(f"pred_{h}.pkl")`, normalize DataFrame→Series (`obj["score"]` for TRA else `obj.iloc[:,0]`), rename to `f"{model_id}_{h}"`.
- `RidgeStacker` (`production/ensemble_stacker.py`): `fit_oof(base_preds, y)`, `predict_with_fallback(base_preds)`.
- Serving current = `backend/app/core/qlib_adapter.py::get_latest_recorder_id(exp)` (newest start_time with loadable pred.pkl).
- Production experiment name = `Settings().retrain_recorder_experiment`; ensemble recorder = `ensemble_<end_date>`.
- Recorder scan for history/compare = `backend/app/evaluation/service.py::list_recorders_with_summary()` enumerates ALL experiments → candidates appear automatically.
- SchedulerManager job is currently fixed to `run-once` (P1); `make_subprocess_retrain_job` + `run_now` + `_gated_job_fn` thread `(job_id, log_path)`.

---

## Task 1: Extract `pool_stack_write` from run_once (refactor, behavior-preserving)

**Files:**
- Modify: `production/rolling_train.py` (extract the ensemble/stack/write/persist block of `run_once` into a module-level function `pool_stack_write(cfg, series_list, end_date, members, *, experiment_name, recorder_name, seed_serving) -> Path`; `run_once` calls it).
- Test: `production/tests/test_pool_stack_write.py`

**Worktree:** `E:\Projects\qlib\.claude\worktrees\sweet-sanderson-214f6a` (branch `feat/training-studio`); full paths + `git -C`; Python `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8`; production tests cwd = worktree root.

**Context:** The ensemble step (concat → stacker → score/consensus/ewma → write pred.pkl → persist recorder) is currently inline in `run_once` (~lines 451–531). Extract it verbatim into a reusable function so `reblend_single` (Task 3) can call the SAME logic but target a different experiment/recorder. This is a behavior-preserving refactor: `run_once` must produce identical output.

- [ ] **Step 1: Write a characterization test for the pure scoring part**

Extract the *pure* part (stacker + score) is hard to isolate from qlib `D.features`. Instead, test that the extracted function exists with the right signature and that a tiny synthetic `series_list` flows through the score/consensus/ewma path without qlib (inject labels). Write:

```python
# production/tests/test_pool_stack_write.py
import pandas as pd
from production.rolling_train import _stack_score  # pure helper extracted below


def test_stack_score_produces_score_column():
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2026-01-05", "2026-01-06"]), ["SH600000", "SZ000001", "SH601318"]],
        names=["datetime", "instrument"],
    )
    base = pd.DataFrame({"lgbm_5d": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}, index=idx)
    labels = pd.Series([0.01, -0.01, 0.02, 0.0, 0.03, -0.02], index=idx, name="y")
    out = _stack_score(base, labels)
    assert "score" in out.columns
    assert len(out) == len(base)
    assert out["score"].notna().any()
```

- [ ] **Step 2: Run to verify it fails**

Run (worktree root): `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest production/tests/test_pool_stack_write.py -v`
Expected: FAIL — `ImportError: cannot import name '_stack_score'`.

- [ ] **Step 3: Extract `_stack_score` + `pool_stack_write`**

In `production/rolling_train.py`, add a pure helper that takes `base_preds` + OOF `labels` and returns the scored frame (factoring the stacker + score/consensus/ewma out of run_once):

```python
def _stack_score(base_preds: pd.DataFrame, oof_labels: pd.Series, *, ewma_alpha: float = 0.5) -> pd.DataFrame:
    """Fit the Ridge stacker on OOF (base_preds rows overlapping oof_labels) and
    score the full base_preds. Pure: no qlib/IO. Mirrors run_once's ensemble math."""
    from production.ensemble_stacker import RidgeStacker
    from production.ensemble_rank_avg import rank_average
    try:
        valid_mask = base_preds.index.get_level_values("datetime").isin(
            oof_labels.index.get_level_values("datetime").unique()
        )
        valid_base = base_preds[valid_mask]
        stacker = RidgeStacker().fit_oof(valid_base, oof_labels)
        unified = stacker.predict_with_fallback(base_preds).rename("score")
    except Exception as exc:
        _log.warning("stacker_failed_using_rank_average error=%s", str(exc))
        unified = (-rank_average(base_preds)).rename("score")
    out = base_preds.copy()
    out["score"] = unified
    out["consensus"] = consensus_per_row(base_preds)
    out = ewma_smooth(out, alpha=ewma_alpha, score_col="score")
    return out
```

Then add `pool_stack_write` that wraps `_stack_score` with the qlib label-fetch + recorder persistence, parameterized by experiment/recorder:

```python
def pool_stack_write(
    cfg: "RollingConfig", series_list: list[pd.Series], end_date: date, members: list[str],
    *, experiment_name: str, recorder_name: str, seed_serving: bool,
) -> Path:
    """Concat base series, refit stacker on the 5d valid window, write pred.pkl,
    and persist a recorder under experiment_name. Used by run_once (production exp)
    and reblend_single (candidate exp)."""
    from qlib.data import D
    from qlib.workflow import R
    base_preds = pd.concat(series_list, axis=1).dropna(how="all")
    # 5d valid-window OOF labels (open-to-open), same as run_once.
    h5 = next(h for h in cfg.horizons if h.name == "5d")
    s_5 = split(end_date=end_date, cfg=h5)
    labels = D.features(instruments=members, fields=["Ref($open, -6) / Ref($open, -1) - 1"],
                        start_time=str(s_5.valid_start), end_time=str(s_5.valid_end))
    labels.columns = ["y"]
    labels.index.names = ["instrument", "datetime"]
    labels = labels.swaplevel("instrument", "datetime").sort_index()["y"]
    out = _stack_score(base_preds, labels, ewma_alpha=cfg.ewma_alpha)
    pred_path = REPO_ROOT / "examples" / "mlruns" / f"pred_{recorder_name}.pkl"
    write_pred_pkl(out, pred_path)
    with R.start(experiment_name=experiment_name, recorder_name=recorder_name):
        try:
            emit_recorder(R.get_recorder().id)
        except Exception:
            pass
        if seed_serving:
            R.save_objects(**{"pred.pkl": out})
    return pred_path
```

> NOTE: This duplicates run_once's ensemble math into reusable form. To keep the refactor behavior-preserving WITHOUT risk, leave `run_once`'s existing inline block AS-IS for this task (do not rewire run_once yet) — `pool_stack_write` is NEW and used only by `reblend_single` (Task 3). (Rewiring run_once to call it is a nice-to-have, NOT required; skipping it avoids regressions.) The `_stack_score` test is the unit gate.

- [ ] **Step 4: Run to verify it passes**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest production/tests/test_pool_stack_write.py production/tests/test_progress_total.py -v`
Expected: PASS (the new `_stack_score` test + existing run_once helper tests unaffected).

- [ ] **Step 5: Commit**

```bash
git -C "E:\Projects\qlib\.claude\worktrees\sweet-sanderson-214f6a" add production/rolling_train.py production/tests/test_pool_stack_write.py
git -C "E:\Projects\qlib\.claude\worktrees\sweet-sanderson-214f6a" commit -m "feat(training): extract _stack_score + pool_stack_write for re-blend reuse"
```

---

## Task 2: `load_prior_model_series` — reuse other models' cached preds

**Files:**
- Create: `production/reblend.py`
- Test: `production/tests/test_reblend_load.py`

**Context:** Mirror `run_split.py:198–237`. Load the full-window `pred_<h>.pkl` for given `model_ids` at a given `end_date` from their `<model>_<h>_<end_date>` recorders, returning named Series. Use a fake recorder list in the test (no qlib).

- [ ] **Step 1: Write the failing test**

```python
# production/tests/test_reblend_load.py
import pandas as pd
from production.reblend import series_from_recorders


class _FakeRec:
    def __init__(self, name, objs):
        self.id = name
        self.info = {"name": name}
        self._objs = objs
    def load_object(self, key):
        return self._objs[key]


def _series(vals):
    idx = pd.MultiIndex.from_product([pd.to_datetime(["2026-01-05"]), ["SH600000", "SZ000001"]],
                                     names=["datetime", "instrument"])
    return pd.Series(vals, index=idx)


def test_series_from_recorders_matches_model_horizon_names():
    recs = [
        _FakeRec("alstm_5d_2026-06-12", {"pred_5d.pkl": _series([0.1, 0.2])}),
        _FakeRec("tra_5d_2026-06-12", {"pred_5d.pkl": pd.DataFrame({"score": _series([0.3, 0.4])})}),
        _FakeRec("lgbm_5d_2026-06-12", {"pred_5d.pkl": _series([0.9, 0.9])}),  # excluded by model_ids
        _FakeRec("ensemble_2026-06-12", {}),  # ignored
    ]
    out = series_from_recorders(recs, end_str="2026-06-12", model_ids=("alstm", "tra"), horizons=("5d",))
    cols = sorted(s.name for s in out)
    assert cols == ["alstm_5d", "tra_5d"]
    # TRA DataFrame normalized to its 'score' column:
    tra = next(s for s in out if s.name == "tra_5d")
    assert list(tra.values) == [0.3, 0.4]


def test_series_from_recorders_empty_when_no_match():
    out = series_from_recorders([], end_str="2026-06-12", model_ids=("alstm",), horizons=("5d",))
    assert out == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest production/tests/test_reblend_load.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'production.reblend'`.

- [ ] **Step 3: Implement**

```python
# production/reblend.py
"""Single-algorithm re-blend: retrain one base model and re-blend it with the
OTHER models' cached predictions (loaded from their existing recorders) into a
CANDIDATE ensemble. See docs/superpowers/plans/2026-06-16-training-studio-p3a-single-algo.md."""
from __future__ import annotations

import logging

import pandas as pd

_log = logging.getLogger("reblend")


def series_from_recorders(recs, *, end_str: str, model_ids, horizons=("1d", "5d", "20d")) -> list[pd.Series]:
    """Load full-window pred_<h>.pkl for the given model_ids at end_str from
    <model>_<h>_<end_str> recorders. Mirrors run_split's pooling load."""
    out: dict[str, pd.Series] = {}
    for rec in recs:
        info = getattr(rec, "info", {})
        run_name = info.get("name") if isinstance(info, dict) else getattr(rec, "name", "")
        for model_id in model_ids:
            for h in horizons:
                if run_name != f"{model_id}_{h}_{end_str}":
                    continue
                try:
                    obj = rec.load_object(f"pred_{h}.pkl")
                except Exception as exc:
                    _log.warning("reblend_load_failed recorder=%s file=pred_%s.pkl error=%s", run_name, h, exc)
                    continue
                s = (obj["score"] if "score" in obj.columns else obj.iloc[:, 0]) if isinstance(obj, pd.DataFrame) else obj
                col = f"{model_id}_{h}"
                if col not in out:
                    out[col] = s.rename(col)
    return list(out.values())
```

- [ ] **Step 4: Run to verify it passes**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest production/tests/test_reblend_load.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git -C "E:\Projects\qlib\.claude\worktrees\sweet-sanderson-214f6a" add production/reblend.py production/tests/test_reblend_load.py
git -C "E:\Projects\qlib\.claude\worktrees\sweet-sanderson-214f6a" commit -m "feat(training): reblend.series_from_recorders (reuse other models' cached preds)"
```

---

## Task 3: `reblend_single` orchestration + CLI `reblend`

**Files:**
- Modify: `production/reblend.py` (add `reblend_single`)
- Modify: `production/rolling_train.py` (add a `reblend` subcommand in `main()`)
- Test: covered by Tasks 1–2 units + the Task 10 live smoke (full orchestration needs qlib; do NOT mock qlib here).

**Context:** `reblend_single(cfg, target_model_id, end_date)`: (1) train the target model for all horizons (reuse `train_lgbm_horizon` / `train_alstm_multihead` / `train_tra_multihead`); (2) load the OTHER enabled models' series via `series_from_recorders` from the production experiment at `end_date`; (3) `pool_stack_write(... experiment_name=<candidate_exp>, recorder_name=f"candidate_{target}_{end_date}", seed_serving=True)`. The candidate experiment = `f"{cfg.experiment_name}_candidates"`.

- [ ] **Step 1: Add `reblend_single` to `production/reblend.py`**

```python
def candidate_experiment_name(prod_experiment: str) -> str:
    return f"{prod_experiment}_candidates"


def reblend_single(cfg, target_model_id: str, end_date):
    """Retrain target_model_id on end_date, reuse the other enabled models'
    cached preds, and write a candidate ensemble recorder. Returns the candidate
    recorder_name. Requires the other models' <model>_<h>_<end_date> recorders to
    exist (i.e. a prior full run for this fold)."""
    from datetime import date as _date
    from qlib.workflow import R
    from production.rolling_train import (
        build_universe, init_qlib, train_lgbm_horizon, pool_stack_write,
    )
    if isinstance(end_date, str):
        end_date = _date.fromisoformat(end_date)
    init_qlib(cfg)
    members, universe_name = build_universe(cfg, end_date)

    enabled = [s["id"] for s in cfg.model_specs if s.get("enabled")]
    if target_model_id not in enabled:
        raise ValueError(f"{target_model_id} not in enabled models {enabled}")
    others = [m for m in enabled if m != target_model_id]

    # 1) Train the target model (all horizons) → series
    series_list = []
    if target_model_id == "lgbm":
        for h in cfg.horizons:
            series_list.append(train_lgbm_horizon(cfg, h, universe_name, end_date))
    elif target_model_id == "alstm":
        from production.train_alstm import train_alstm_multihead
        series_list.extend(train_alstm_multihead(cfg, universe_name, end_date))
    elif target_model_id == "tra":
        from production.train_tra import train_tra_multihead
        series_list.extend(train_tra_multihead(cfg, universe_name, end_date))
    else:
        raise ValueError(f"unknown model {target_model_id}")

    # 2) Reuse the other models' cached preds for the SAME fold
    recs = list(R.list_recorders(experiment_name=cfg.experiment_name).values())
    reused = series_from_recorders(recs, end_str=end_date.isoformat(), model_ids=tuple(others))
    if others and not reused:
        raise RuntimeError(
            f"no cached recorders for {others} at end_date={end_date} — run a full retrain for this fold first"
        )
    series_list.extend(reused)

    # 3) Stack + write to the CANDIDATE experiment (does NOT auto-go-live)
    cand_exp = candidate_experiment_name(cfg.experiment_name)
    recorder_name = f"candidate_{target_model_id}_{end_date.isoformat()}"
    pool_stack_write(cfg, series_list, end_date, members,
                     experiment_name=cand_exp, recorder_name=recorder_name, seed_serving=True)
    _log.info("reblend_candidate_written recorder=%s exp=%s", recorder_name, cand_exp)
    return recorder_name
```

- [ ] **Step 2: Add the `reblend` CLI subcommand in rolling_train.main()**

In `production/rolling_train.py::main()`, add a subparser (next to `run-once`/`backfill`):

```python
    p_reblend = sub.add_parser("reblend", help="Retrain ONE model on the latest fold + re-blend into a candidate.")
    p_reblend.add_argument("--only", required=True, help="model id: lgbm | alstm | tra")
    p_reblend.add_argument("--end-date", default=None, help="fold end date; default = latest ensemble recorder's date")
    p_reblend.add_argument("--config", default="production/configs/rolling_ensemble.yaml")
```

And handle it in the dispatch:

```python
    elif args.cmd == "reblend":
        cfg = load_config(REPO_ROOT / args.config)
        from production.reblend import reblend_single, latest_full_fold_end_date
        end = date.fromisoformat(args.end_date) if args.end_date else latest_full_fold_end_date(cfg)
        name = reblend_single(cfg, args.only, end)
        print(f"OK: wrote candidate {name}")
```

Add `latest_full_fold_end_date(cfg)` to `reblend.py` (finds the newest `ensemble_<date>` recorder's date in the production experiment):

```python
def latest_full_fold_end_date(cfg):
    from datetime import date as _date
    from qlib.workflow import R
    recs = R.list_recorders(experiment_name=cfg.experiment_name)
    dates = []
    for rec in recs.values():
        nm = (rec.info or {}).get("name", "") if isinstance(rec.info, dict) else getattr(rec, "name", "")
        if nm.startswith("ensemble_"):
            try:
                dates.append(_date.fromisoformat(nm[len("ensemble_"):]))
            except ValueError:
                pass
    if not dates:
        raise RuntimeError("no ensemble_<date> recorder found — run a full retrain first")
    return max(dates)
```

- [ ] **Step 3: Smoke-import check (no full run here)**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -c "import production.reblend; import production.rolling_train; print('ok')"` (worktree root)
Expected: `ok` (imports resolve; no circular import — `reblend` imports rolling_train lazily inside functions).

- [ ] **Step 4: Re-run the unit tests (no regression)**

Run: `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest production/tests/test_reblend_load.py production/tests/test_pool_stack_write.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C "E:\Projects\qlib\.claude\worktrees\sweet-sanderson-214f6a" add production/reblend.py production/rolling_train.py
git -C "E:\Projects\qlib\.claude\worktrees\sweet-sanderson-214f6a" commit -m "feat(training): reblend_single + 'reblend --only X' CLI (candidate experiment)"
```

---

## Task 4: SchedulerManager — parametrized run args (full vs single-algo)

**Files:**
- Modify: `backend/app/scheduling/service.py` (`make_subprocess_retrain_job` builds argv from per-run params; `run_now(extra_args)`; thread through `_gated_job_fn`).
- Test: `backend/app/scheduling/tests/test_run_args.py`

**Context:** Today the job runs fixed `run-once`. For single-algo, the job must run `reblend --only <model>`. Thread an optional `run_spec` (argv tail) from `run_now` → `_gated_job_fn` → the job fn. Keep `_session_maker=None`-guarded persistence (P2) intact.

- [ ] **Step 1: Write the failing test**

```python
# backend/app/scheduling/tests/test_run_args.py
from pathlib import Path
import pytest
from app.scheduling.service import make_subprocess_retrain_job


def test_job_builds_reblend_argv(monkeypatch):
    captured = {}
    async def fake_exec(*args, **kwargs):
        captured["args"] = args
        class _P:
            stdout = None
            async def wait(self): return 0
        return _P()
    import app.scheduling.service as svc
    monkeypatch.setattr(svc.asyncio, "create_subprocess_exec", fake_exec)
    job = make_subprocess_retrain_job(python_path="py", repo_root=Path("."))
    import asyncio
    # full run → run-once
    asyncio.get_event_loop().run_until_complete(job("j1", Path("x.log"), None))
    assert "run-once" in captured["args"]
    # single-algo → reblend --only lgbm
    asyncio.get_event_loop().run_until_complete(job("j2", Path("y.log"), ["reblend", "--only", "lgbm"]))
    assert "reblend" in captured["args"] and "lgbm" in captured["args"]
```

> If the existing job fn raises on `proc.stdout is None`, the test's `_P.stdout=None` path must still let argv be captured before the raise — the `create_subprocess_exec` call (and thus `captured`) happens first, so the assertion holds even if the fn then raises; wrap the `run_until_complete` in `pytest.raises(Exception)` if needed, or set `_P.stdout` to an async-iterable empty stream. Adjust the fake to match the real drain loop so the assertion is reached.

- [ ] **Step 2: Run to verify it fails**

Run (cwd backend): `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest app/scheduling/tests/test_run_args.py -v`
Expected: FAIL — job fn signature is `(job_id, log_path)`, not `(job_id, log_path, run_spec)`.

- [ ] **Step 3: Parametrize the job + run_now**

In `make_subprocess_retrain_job`, change the inner job to accept `run_spec: list[str] | None` and build argv:

```python
    async def _job(job_id: str, log_path: Path, run_spec: list[str] | None = None) -> None:
        argv = run_spec if run_spec else ["run-once"]
        ...
        proc = await asyncio.create_subprocess_exec(
            python_path, "-m", "production.rolling_train", *argv,
            cwd=str(repo_root), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        ... (rest unchanged: write per-job log, raise on non-zero)
```

Update `JobCallable = Callable[[str, Path, "list[str] | None"], Awaitable[None]]`.

In `run_now`, accept and store the spec:

```python
    async def run_now(self, session, force=False, run_spec: list[str] | None = None) -> str:
        ...
        self._remember_job(job_id, { ... , "run_spec": run_spec })
        ...
```

In `_gated_job_fn`, pass it: `await self._raw_job_fn(_tracked_job_id, self._log_path_for(_tracked_job_id), (entry or {}).get("run_spec"))`. The cron-mint entry uses `run_spec=None` (full retrain).

- [ ] **Step 4: Run to verify it passes + scheduling regression**

Run (cwd backend): `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest app/scheduling/ -v`
Expected: PASS (new test + all P1/P2 scheduling tests; the P1 `test_job_log_path.py` fake jobs take `(job_id, log_path)` — UPDATE those fakes to accept an optional 3rd arg `run_spec=None`, or the call passes 3 args; adjust the P1 test fakes' signatures to `(job_id, log_path, run_spec=None)`).

- [ ] **Step 5: Commit**

```bash
git -C "E:\Projects\qlib\.claude\worktrees\sweet-sanderson-214f6a" add backend/app/scheduling/service.py backend/app/scheduling/tests/
git -C "E:\Projects\qlib\.claude\worktrees\sweet-sanderson-214f6a" commit -m "feat(training): SchedulerManager run_spec — parametrized full vs reblend argv"
```

---

## Task 5: Backend — single-algo trigger via /api/training/run

**Files:**
- Modify: `backend/app/training/schemas.py` (`TrainRequest` already has `scope`/`models` reserved — wire them).
- Modify: `backend/app/training/router.py` (`run_training` builds `run_spec` for scope=single).
- Modify: `backend/app/scheduling/service.py` (`_persist_run(phase="start")` records scope/models from the run_spec/entry).
- Test: `backend/app/training/tests/test_single_algo_trigger.py`

**Context:** `POST /api/training/run {scope:"single", models:["lgbm"]}` → `run_spec = ["reblend", "--only", "lgbm"]` → `manager.run_now(..., run_spec=...)`. Record `scope="single"`, `models_json=["lgbm"]` in training_runs.

- [ ] **Step 1: Write the failing test**

```python
# backend/app/training/tests/test_single_algo_trigger.py
import pytest
from httpx import ASGITransport, AsyncClient
from app.main import create_app
from app.scheduling.router import set_manager
from app.core.db import get_session


class _Mgr:
    def __init__(self): self.spec = "UNSET"
    async def run_now(self, session, force=False, run_spec=None):
        self.spec = run_spec
        return "jX"
    def get_job_status(self, jid): return None
    def get_active_job(self): return None


@pytest.mark.asyncio
async def test_single_algo_builds_reblend_spec():
    app = create_app(); m = _Mgr(); set_manager(m)
    app.dependency_overrides[get_session] = lambda: None
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        r = await ac.post("/api/training/run", json={"scope": "single", "models": ["lgbm"], "force": True})
        assert r.status_code == 200 and r.json()["job_id"] == "jX"
    assert m.spec == ["reblend", "--only", "lgbm"]


@pytest.mark.asyncio
async def test_full_scope_uses_none_spec():
    app = create_app(); m = _Mgr(); set_manager(m)
    app.dependency_overrides[get_session] = lambda: None
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
        await ac.post("/api/training/run", json={"scope": "full", "force": True})
    assert m.spec is None
```

- [ ] **Step 2: Run to verify it fails**

Run (cwd backend): `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest app/training/tests/test_single_algo_trigger.py -v`
Expected: FAIL — run_training ignores scope/models (passes no run_spec).

- [ ] **Step 3: Wire scope→run_spec in the router**

In `backend/app/training/schemas.py`, give `TrainRequest` a `models` field:

```python
class TrainRequest(BaseModel):
    scope: str = Field("full", description='"full" | "single"')
    models: list[str] = Field(default_factory=list, description='for scope="single": exactly one model id')
    force: bool = Field(False)
```

In `backend/app/training/router.py::run_training`:

```python
@router.post("/run")
async def run_training(payload: TrainRequest, session: AsyncSession = Depends(get_session)):
    run_spec = None
    if payload.scope == "single":
        if len(payload.models) != 1:
            from app.core.exceptions import BusinessError
            raise BusinessError("scope=single requires exactly one model", code="bad_single_models")
        run_spec = ["reblend", "--only", payload.models[0]]
    try:
        job_id = await get_manager().run_now(session, force=payload.force, run_spec=run_spec)
        return {"status": "started", "job_id": job_id}
    except TradingHoursViolation as exc:
        return {"status": "rejected", "reason": str(exc)}
    except AlreadyRunning as exc:
        return {"status": "rejected", "reason": str(exc)}
```

- [ ] **Step 4: Record scope/models in training_runs**

In `backend/app/scheduling/service.py::_persist_run`, derive scope/models from the entry's `run_spec` for `phase="start"`:

```python
                if phase == "start":
                    from app.training import store
                    rs = (entry or {}).get("run_spec")
                    if rs and "reblend" in rs and "--only" in rs:
                        scope, models = "single", [rs[rs.index("--only") + 1]]
                    else:
                        scope, models = "full", None
                    await store.record_run_start(session, job_id=job_id,
                        kind=(entry or {}).get("kind", "manual"), scope=scope, models=models, started_at=now)
```

- [ ] **Step 5: Run to verify it passes**

Run (cwd backend): `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest app/training/tests/test_single_algo_trigger.py app/training/ -v`
Expected: PASS (new tests + all prior training tests).

- [ ] **Step 6: Commit**

```bash
git -C "E:\Projects\qlib\.claude\worktrees\sweet-sanderson-214f6a" add backend/app/training/schemas.py backend/app/training/router.py backend/app/scheduling/service.py backend/app/training/tests/test_single_algo_trigger.py
git -C "E:\Projects\qlib\.claude\worktrees\sweet-sanderson-214f6a" commit -m "feat(training): /api/training/run scope=single → reblend spec + records scope/models"
```

---

## Task 6: Promote endpoint — candidate → production

**Files:**
- Modify: `backend/app/models/service.py` (add `promote_candidate(recorder_id, candidate_experiment)`)
- Modify: `backend/app/training/router.py` (add `POST /api/training/promote`)
- Modify: `backend/app/training/schemas.py` (add `PromoteRequest`)
- Test: `backend/app/models/tests/test_promote.py`

**Context:** Promote = load the chosen candidate recorder's `pred.pkl` → write it into a NEW recorder in the production experiment (`recorder_name=f"promoted_{candidate_name}"`) → it becomes newest with pred.pkl → `get_latest_recorder_id` serves it. mlflow-correct (uses `R.start`/`save_objects`); no directory surgery.

- [ ] **Step 1: Write the failing test**

```python
# backend/app/models/tests/test_promote.py
import pandas as pd
import pytest
from app.models import service as svc


def test_promote_copies_pred_into_production(monkeypatch):
    saved = {}
    fake_pred = pd.DataFrame({"score": [1.0, 2.0]})

    class _Rec:
        def load_object(self, key): return fake_pred
    class _Started:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    class _R:
        @staticmethod
        def get_recorder(recorder_id=None, experiment_name=None): return _Rec()
        @staticmethod
        def start(experiment_name=None, recorder_name=None):
            saved["exp"] = experiment_name; saved["name"] = recorder_name; return _Started()
        @staticmethod
        def save_objects(**kw): saved["pred_rows"] = len(kw["pred.pkl"])
    monkeypatch.setattr(svc, "R", _R, raising=False)
    monkeypatch.setattr(svc, "init_qlib_once", lambda *a, **k: None, raising=False)

    out = svc.promote_candidate("cand_rec_1", candidate_experiment="exp_candidates", production_experiment="exp")
    assert out["status"] == "promoted"
    assert saved["exp"] == "exp"
    assert saved["name"].startswith("promoted_")
    assert saved["pred_rows"] == 2
```

- [ ] **Step 2: Run to verify it fails**

Run (cwd backend): `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest app/models/tests/test_promote.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'promote_candidate'`.

- [ ] **Step 3: Implement `promote_candidate`**

In `backend/app/models/service.py` (ensure `from qlib.workflow import R` + `init_qlib_once` are importable at module scope as the test monkeypatches `svc.R`/`svc.init_qlib_once` — match how other functions import; if they import locally, refactor the test to patch the local import OR add module-level `R`):

```python
def promote_candidate(recorder_id: str, *, candidate_experiment: str, production_experiment: str | None = None) -> dict:
    """Load a candidate recorder's pred.pkl and save it as a NEW recorder in the
    production experiment, making it the newest (served) model. Non-destructive."""
    from datetime import datetime, timezone
    init_qlib_once()
    from app.core.config import Settings
    prod = production_experiment or Settings().retrain_recorder_experiment
    rec = R.get_recorder(recorder_id=recorder_id, experiment_name=candidate_experiment)
    pred = rec.load_object("pred.pkl")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    name = f"promoted_{recorder_id[:8]}_{ts}"
    with R.start(experiment_name=prod, recorder_name=name):
        R.save_objects(**{"pred.pkl": pred})
    return {"status": "promoted", "production_experiment": prod, "new_recorder_name": name, "from_recorder_id": recorder_id}
```

- [ ] **Step 4: Add the endpoint + schema**

`backend/app/training/schemas.py`:

```python
class PromoteRequest(BaseModel):
    recorder_id: str
    candidate_experiment: str
```

`backend/app/training/router.py`:

```python
@router.post("/promote")
def promote(payload: PromoteRequest):
    from app.models.service import promote_candidate
    return promote_candidate(payload.recorder_id, candidate_experiment=payload.candidate_experiment)
```

- [ ] **Step 5: Run to verify it passes**

Run (cwd backend): `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest app/models/tests/test_promote.py app/training/ -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git -C "E:\Projects\qlib\.claude\worktrees\sweet-sanderson-214f6a" add backend/app/models/service.py backend/app/training/router.py backend/app/training/schemas.py backend/app/models/tests/test_promote.py
git -C "E:\Projects\qlib\.claude\worktrees\sweet-sanderson-214f6a" commit -m "feat(training): POST /api/training/promote — copy candidate pred.pkl into production recorder"
```

---

## Task 7: History exposes candidate experiment + scope/candidate flag

**Files:**
- Modify: `backend/app/training/service.py::build_history` (mark candidate recorders so the UI can show a Promote action).
- Test: `backend/app/training/tests/test_history_endpoint.py` (extend)

**Context:** `list_recorders_with_summary` already enumerates ALL experiments, so candidate recorders appear. Add an `experiment` + `is_candidate` field to `TrainingRunRow` (true when `experiment` endswith `_candidates` or `run_name` startswith `candidate_`) so the frontend renders Promote only for candidates.

- [ ] **Step 1: Extend the test** — add a recorder with `run_name="candidate_lgbm_2026-06-16"`, assert its row has `is_candidate True`; a normal `ensemble_*` row has `is_candidate False`. (Mirror the existing `test_history_endpoint.py` structure.)

- [ ] **Step 2: Run → fails.** (field missing)

- [ ] **Step 3: Add `experiment: str | None` + `is_candidate: bool = False` to `TrainingRunRow`; set in `build_history`** (both the linked-run branch and the historical branch): `is_candidate = bool(rec and (str(getattr(rec,"experiment","")).endswith("_candidates") or str(getattr(rec,"run_name","")).startswith("candidate_")))`. For run rows without a rec, default False.

- [ ] **Step 4: Run → passes.**

- [ ] **Step 5: Commit** `feat(training): mark candidate recorders in history (is_candidate)`

---

## Task 8: Frontend — single-algo trigger + candidate types

**Files:**
- Modify: `frontend/src/api/client.ts` (`training.run` accepts `{scope, models}`; add `training.promote`; extend `TrainingRunRow` with `experiment`/`is_candidate`).
- Modify: `frontend/src/training/hooks.ts` (`useStartTraining` accepts `{scope, models}`; add `usePromote`).

- [ ] **Step 1:** Update `api.training.run` to `(body: { scope: 'full'|'single'; models?: string[]; force?: boolean })`; add `promote: (recorder_id, candidate_experiment) => POST /api/training/promote`. Add `experiment: string|null` + `is_candidate: boolean` to `TrainingRunRow`. Update `useStartTraining` mutationFn to take the body; add `usePromote` (invalidates `['training','runs']`).
- [ ] **Step 2:** `npm run build` green.
- [ ] **Step 3:** Commit `feat(training): frontend single-algo run + promote client/hooks`

(Full code mirrors the P1/P2 client+hooks patterns; the implementer reads the existing `training` group + hooks and extends them.)

---

## Task 9: Frontend — 训练 single-algo选择 + Promote button

**Files:**
- Modify: `frontend/src/pages/Training.tsx`

- [ ] **Step 1:** In the 训练 section, add a small control: a radio/segmented "全量 / 单算法", and when 单算法, a select of `lgbm|alstm|tra`. The 立即训练 button calls `start.mutate({scope, models})` accordingly. In the 历史模型 table, for rows with `is_candidate`, render a small "设为当前" (promote) button calling `usePromote().mutate({recorder_id, candidate_experiment: row.experiment})` with a `confirm` + toast (mirror the rollback button).
- [ ] **Step 2:** `npm run build` green.
- [ ] **Step 3:** Commit `feat(training): 训练工作台 单算法训练选择 + 候选 promote 按钮`

---

## Task 10: Regression + build + live smoke

- [ ] **Step 1:** Backend: `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest app/training/ app/scheduling/ app/models/tests/test_promote.py -v` (cwd backend) → all green.
- [ ] **Step 2:** Production units: `F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m pytest production/tests/test_reblend_load.py production/tests/test_pool_stack_write.py -v` (worktree root) → green.
- [ ] **Step 3:** Frontend: `npm run build` (cwd frontend) → green.
- [ ] **Step 4 (live smoke, data-bearing env):** with a prior full run present, run `python -m production.rolling_train reblend --only lgbm` → confirm it (a) does NOT retrain alstm/tra (logs show only LGBM training + "reblend_candidate_written"), (b) writes `candidate_lgbm_<date>` in the `<exp>_candidates` experiment, (c) does NOT change the served model. Then in the UI: 训练→单算法→LGBM→立即训练; the candidate appears in 历史 (is_candidate); compare it vs current; 设为当前 → confirm the served model switches. 
- [ ] **Step 5:** Commit any smoke fixes.

---

## Self-Review (completed during authoring)
- **Spec coverage:** single-algo re-blend (T1–3), trigger (T4–5), promote (T6), history surface (T7), frontend (T8–9), verify (T10). ✓
- **Re-blend correctness:** reuses run_once's exact stacker math via `_stack_score`/`pool_stack_write`; reuses other models via the run_split-proven load; same end_date ⇒ deterministic fold. Refit on cached OOF (the confirmed choice). ✓
- **Safety:** candidates → separate `_candidates` experiment ⇒ serving (`get_latest_recorder_id` on production exp) ignores them; promote is non-destructive (new recorder, no dir surgery). ✓
- **Placeholder scan:** novel logic (`_stack_score`, `series_from_recorders`, `reblend_single`, `promote_candidate`, run_spec threading) has complete code; T8/T9 reference the established client/hooks/page patterns with explicit edits (not placeholders). ✓
- **Type/contract consistency:** `run_spec` threaded run_now→_gated_job_fn→job fn; `scope/models` ↔ TrainRequest ↔ training_runs; `is_candidate`/`experiment` ↔ TrainingRunRow (backend+frontend). ✓
- **Risks:** T1 duplicates ensemble math (mitigated: run_once left intact; `_stack_score` unit-tested). T4 changes the job-fn signature again (mitigated: update P1 test fakes). T6 monkeypatch needs module-level `R`/`init_qlib_once` in models/service (the implementer verifies/adjusts the import site). The re-blend full path is qlib-heavy ⇒ validated by the T10 live smoke, not unit mocks.
- **Deferred (later P3 slices):** registry enable-toggles (P3b), loss curves (P3c), created_at unification (P3d).
