# Model Evaluation Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable evaluation pipeline (CLI + REST API + Web UI) that takes any qlib recorder, computes the 8-metric scorecard + multi-regime breakdown + acceptance pass/fail, lets users compare two recorders via paired t-test, and surfaces the diagnosis on a `/evaluation` page in the companion app.

**Architecture:** Backend `app/evaluation/` module wraps the existing `production/metrics.py` helpers (`compute_scorecard`, `regime_split`, `paired_ttest`) + `production/validate_acceptance.check_acceptance`. New service loads a recorder's `pred.pkl`, fetches matching open-to-open labels via qlib `D.features`, runs the metric suite, and caches the result via `functools.lru_cache(maxsize=32)` keyed by recorder_id (no DB persistence in v1; recorder predictions are immutable so cache is correct). REST endpoints at `/api/evaluation/*` are consumed by a new React `/evaluation` page (list + detail + compare modes). Same service powers a `python -m production.eval_recorder <id>` CLI that writes JSON + Markdown reports.

**Tech Stack:** Python 3.10 + qlib (`F:/Tools/Anaconda/envs/qlib/python.exe`); FastAPI + SQLAlchemy async (we don't add DB tables — read-only on existing mlruns); React 18 + TanStack Query + Vite + Tailwind; openapi-typescript codegen.

**Spec source:** Spec §8 (scorecard + regime + significance) + §11 (acceptance thresholds) of `docs/superpowers/specs/2026-05-21-rolling-ensemble-algorithm-design.md`.

**Run from main repo root** `E:/Projects/qlib`. Python: `F:/Tools/Anaconda/envs/qlib/python.exe`. Backend tests from `backend/` cwd. Frontend typecheck from `frontend/` cwd. Backend is on :8000, frontend on :5173 (dev servers already running).

---

## File Structure

### Backend (`backend/app/evaluation/`)
- `__init__.py` — package marker
- `schemas.py` — Pydantic models: `RecorderSummary`, `ScorecardData`, `AcceptanceResult`, `RegimeMetrics`, `EvalResult`, `CompareResult`, `EvalRunRequest`
- `service.py` — pure logic: `list_recorders_with_summary`, `evaluate_recorder` (cached), `compare_recorders`, `_canonical_label_series`, `_overlapping_regimes`
- `router.py` — 4 endpoints: `GET /recorders`, `GET /results/{recorder_id}`, `POST /run`, `GET /compare`
- `tests/__init__.py`
- `tests/test_service.py` — pure-function tests with mocked qlib
- `tests/test_router.py` — FastAPI TestClient end-to-end

### Backend integration
- `backend/app/main.py` — register `evaluation_router` at `/api/evaluation`

### CLI (`production/`)
- `eval_recorder.py` — argparse wrapper that imports the backend service, writes JSON + Markdown to `production/reports/<recorder_id>_<timestamp>.{json,md}`

### Frontend (`frontend/src/`)
- `pages/Evaluation.tsx` — top-level page; tabbed view (List / Compare)
- `pages/evaluation/types.ts` — local type aliases mirroring backend schema
- `pages/evaluation/hooks.ts` — `useRecorders`, `useEvaluation`, `useCompare`
- `pages/evaluation/RecorderRow.tsx` — one row in the list; clickable to open detail
- `pages/evaluation/ScorecardCard.tsx` — 8 metric cells with threshold-based color (green/red)
- `pages/evaluation/RegimeChart.tsx` — small 5-bar chart of per-regime IR
- `pages/evaluation/AcceptanceLights.tsx` — 5 dots (one per criterion); green pass / red fail
- `pages/evaluation/CompareView.tsx` — side-by-side scorecards + paired t-test result
- `api/client.ts` — add `evaluation` namespace
- `api/types.gen.ts` — regenerated
- `App.tsx` — add `/evaluation` route
- `components/Layout.tsx` — add "Evaluation" nav link

---

## Execution Order

```
Phase A (backend)        T1 → T2 → T3 → T4 → T5
Phase B (CLI)            T6                          ← ships independently from frontend
Phase C (frontend)       T7 → T8 → T9 → T10
Phase D (verify+report)  T11                         ← runs eval on daily_cn_fresh, writes optimization plan
```

Each phase end produces a working slice: T5 = backend complete; T6 = CLI works; T10 = UI live; T11 = diagnostic output for the user.

---

## Phase A — Backend

### Task 1: `evaluation` module skeleton + schemas

**Files:**
- Create: `backend/app/evaluation/__init__.py`
- Create: `backend/app/evaluation/schemas.py`

- [ ] **Step 1: Create package marker**

`backend/app/evaluation/__init__.py`:

```python
"""Recorder evaluation pipeline — computes 8-metric scorecard, regime
breakdown, acceptance pass/fail, paired t-test comparison.

Wraps the existing production/metrics.py + production/validate_acceptance.py
helpers and exposes them via REST + a CLI (production/eval_recorder.py).
"""
```

- [ ] **Step 2: Define all schemas**

`backend/app/evaluation/schemas.py`:

```python
from __future__ import annotations

from pydantic import BaseModel, Field


class ScorecardData(BaseModel):
    """The 8-metric scorecard (per spec §8).

    All metrics computed against open-to-open returns + TopK=30 long-only
    portfolio with `bps` transaction cost adjustment.
    """
    ic_mean: float
    ric_mean: float
    icir: float
    top_bottom_spread_monthly: float    # in percent (e.g. 1.8 = 1.8%/month)
    annual_excess_return: float         # decimal (0.15 = +15%)
    ir: float
    max_drawdown: float                 # negative number (-0.12 = -12%)
    daily_turnover: float               # decimal (0.18 = 18%)


class AcceptanceResult(BaseModel):
    """Per-criterion pass/fail against spec §11 thresholds."""
    passed: bool
    details: dict[str, bool]


class RegimeMetrics(BaseModel):
    """One regime segment's scorecard + label (e.g. '2020-COVID')."""
    label: str
    start: str                          # ISO date
    end: str                            # ISO date
    sample_size: int                    # number of (date, symbol) pairs evaluated
    scorecard: ScorecardData


class RecorderSummary(BaseModel):
    """Lightweight summary for the list view. Computed without full eval."""
    recorder_id: str
    experiment: str
    run_name: str
    created_at: str                     # ISO timestamp
    pred_start: str | None = None       # earliest prediction date
    pred_end: str | None = None         # latest prediction date
    pred_rows: int | None = None        # total rows in pred.pkl
    has_eval: bool = False              # True if cache has a result for this recorder
    # Lightweight 'quick look' metrics when has_eval=True. Detail view fetches the full scorecard.
    ic_mean: float | None = None
    ir: float | None = None
    acceptance_passed: bool | None = None


class EvalResult(BaseModel):
    """Full evaluation result for one recorder."""
    recorder_id: str
    experiment: str
    run_name: str
    computed_at: str                    # ISO timestamp
    window_start: str                   # ISO date of earliest evaluated prediction
    window_end: str                     # ISO date of latest evaluated prediction
    sample_size: int                    # rows after label join
    top_k: int                          # portfolio TopK used
    cost_bps: float                     # cost adjustment used
    scorecard: ScorecardData
    regimes: list[RegimeMetrics]
    acceptance: AcceptanceResult


class CompareResult(BaseModel):
    """Side-by-side comparison of two recorders + paired t-test on daily IC."""
    a: EvalResult
    b: EvalResult
    paired_t_stat: float
    paired_p_value: float
    significant_at_05: bool             # True iff p < 0.05
    ic_delta: float                     # b.scorecard.ic_mean - a.scorecard.ic_mean
    ir_delta: float                     # b.scorecard.ir - a.scorecard.ir
    verdict: str                        # 'b significantly better', 'a significantly better', 'no significant difference'


class EvalRunRequest(BaseModel):
    recorder_id: str
    top_k: int = Field(default=30, ge=1, le=300)
    cost_bps: float = Field(default=10, ge=0)
    force_refresh: bool = False         # if True, bypass cache and recompute
```

- [ ] **Step 3: Commit**

```
cd E:/Projects/qlib
git add backend/app/evaluation/__init__.py backend/app/evaluation/schemas.py
git commit -m "feat(evaluation): module skeleton + Pydantic schemas"
```

---

### Task 2: Service — list recorders with summary

**Files:**
- Create: `backend/app/evaluation/service.py` (just the list function for now)
- Create: `backend/app/evaluation/tests/__init__.py`
- Create: `backend/app/evaluation/tests/test_service.py` (first test)

- [ ] **Step 1: Write the failing test for `list_recorders_with_summary`**

`backend/app/evaluation/tests/__init__.py`: empty.

`backend/app/evaluation/tests/test_service.py`:

```python
import pytest

from app.evaluation.service import list_recorders_with_summary
from app.core.qlib_adapter import init_qlib_once


@pytest.fixture(scope="module")
def qlib_ready():
    try:
        init_qlib_once()
    except Exception as exc:
        pytest.skip(f"qlib not initializable: {exc}")


def test_list_recorders_returns_at_least_one_known(qlib_ready):
    summaries = list_recorders_with_summary()
    # The dev environment must have at least the daily_cn_fresh recorder
    assert any(s.experiment == "daily_cn_fresh" for s in summaries), \
        "expected daily_cn_fresh experiment to have at least one recorder"


def test_summary_fields_are_populated(qlib_ready):
    summaries = list_recorders_with_summary()
    if not summaries:
        pytest.skip("no recorders available")
    s = summaries[0]
    assert s.recorder_id
    assert s.experiment
    assert s.run_name
    assert s.created_at
    # pred_start/end/rows may be None on errors but should usually be set
    if s.pred_rows is not None:
        assert s.pred_rows > 0
    assert s.has_eval is False  # cache empty on first call
```

- [ ] **Step 2: Run — expect failure**

```
cd E:/Projects/qlib/backend
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/evaluation/tests/test_service.py::test_list_recorders_returns_at_least_one_known -v
```

Expected: `ModuleNotFoundError: No module named 'app.evaluation.service'`.

- [ ] **Step 3: Implement `list_recorders_with_summary`**

`backend/app/evaluation/service.py`:

```python
"""Evaluation service — load recorders, compute scorecards, compare.

All metric math is delegated to production/metrics.py + production/validate_acceptance.py
(the rolling_train pipeline uses the same helpers, so eval numbers always
match what was computed at train time).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from app.core.config import Settings
from app.core.qlib_adapter import init_qlib_once
from app.evaluation.schemas import RecorderSummary

# Add the repo root so we can import production.metrics / production.validate_acceptance
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))


def list_recorders_with_summary() -> list[RecorderSummary]:
    """Enumerate all qlib recorders across all experiments and return a
    lightweight summary for each. Cheap — does NOT load pred.pkl."""
    init_qlib_once()
    from qlib.workflow import R

    out: list[RecorderSummary] = []
    # List experiments (excludes the deleted-trash sentinel)
    exp_ids = _list_experiment_ids()
    for exp_id in exp_ids:
        exp_name = _experiment_name(exp_id)
        if exp_name in (None, "Default"):
            continue
        try:
            recs = R.list_recorders(experiment_name=exp_name)
        except Exception:
            continue
        for rec_id, rec in recs.items():
            info = rec.info or {}
            run_name = info.get("name", rec_id[:8])
            start_ms = info.get("start_time", 0)
            created_at = (
                pd.to_datetime(start_ms, unit="ms", utc=True).isoformat()
                if start_ms
                else ""
            )
            # Stat pred.pkl size cheaply (without loading the dataframe)
            pred_start, pred_end, pred_rows = _peek_pred_pkl(rec_id, exp_name)
            cached = _cache_has(rec_id)
            quick = _cache_quick_look(rec_id) if cached else (None, None, None)
            out.append(
                RecorderSummary(
                    recorder_id=rec_id,
                    experiment=exp_name,
                    run_name=run_name,
                    created_at=created_at,
                    pred_start=pred_start,
                    pred_end=pred_end,
                    pred_rows=pred_rows,
                    has_eval=cached,
                    ic_mean=quick[0],
                    ir=quick[1],
                    acceptance_passed=quick[2],
                )
            )

    # Sort by created_at desc (newest first)
    out.sort(key=lambda s: s.created_at, reverse=True)
    return out


def _list_experiment_ids() -> list[str]:
    """Walk <mlruns_root>/<exp_id>/ and return all valid experiment dir names."""
    settings = Settings()
    root = settings.mlruns_path
    if not root.exists():
        return []
    out = []
    for d in root.iterdir():
        if d.is_dir() and (d / "meta.yaml").exists() and d.name != ".trash":
            out.append(d.name)
    return out


def _experiment_name(exp_id: str) -> str | None:
    """Read mlruns/<exp_id>/meta.yaml and return the experiment name."""
    settings = Settings()
    meta = settings.mlruns_path / exp_id / "meta.yaml"
    if not meta.exists():
        return None
    for line in meta.read_text(encoding="utf-8").splitlines():
        if line.startswith("name:"):
            return line.split(":", 1)[1].strip()
    return None


def _peek_pred_pkl(rec_id: str, exp_name: str) -> tuple[str | None, str | None, int | None]:
    """Load the recorder's pred.pkl just enough to return date range + row count.
    Returns (None, None, None) on missing/unreadable files."""
    settings = Settings()
    # Find the artifact path. qlib mlflow layout: <mlruns_root>/<exp_id>/<rec_id>/artifacts/
    exp_id = _find_exp_id_for_name(exp_name)
    if exp_id is None:
        return None, None, None
    artifacts = settings.mlruns_path / exp_id / rec_id / "artifacts"
    # Most common name is pred.pkl; for sub-horizon recorders it's pred_<N>.pkl
    candidates = [
        artifacts / "pred.pkl",
        artifacts / "pred_1d.pkl",
        artifacts / "pred_5d.pkl",
        artifacts / "pred_20d.pkl",
    ]
    for c in candidates:
        if c.exists():
            try:
                df = pd.read_pickle(c)
                if isinstance(df, pd.Series):
                    df = df.to_frame()
                dates = df.index.get_level_values(0)
                return (
                    str(pd.Timestamp(dates.min()).date()),
                    str(pd.Timestamp(dates.max()).date()),
                    int(df.shape[0]),
                )
            except Exception:
                continue
    return None, None, None


def _find_exp_id_for_name(exp_name: str) -> str | None:
    for exp_id in _list_experiment_ids():
        if _experiment_name(exp_id) == exp_name:
            return exp_id
    return None


# Cache helpers — wired in Task 3 once evaluate_recorder exists.
def _cache_has(recorder_id: str) -> bool:
    """Return True iff evaluate_recorder has been called for this recorder
    since process startup."""
    return False


def _cache_quick_look(recorder_id: str) -> tuple[float | None, float | None, bool | None]:
    """Pull (ic_mean, ir, acceptance_passed) from cache if present."""
    return (None, None, None)
```

- [ ] **Step 4: Run tests — expect PASS**

```
cd E:/Projects/qlib/backend
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/evaluation/tests/test_service.py -v
```

Expected: 2 tests PASS. (qlib loads, daily_cn_fresh recorder discovered.)

- [ ] **Step 5: Commit**

```
cd E:/Projects/qlib
git add backend/app/evaluation/service.py backend/app/evaluation/tests/__init__.py backend/app/evaluation/tests/test_service.py
git commit -m "feat(evaluation): list_recorders_with_summary"
```

---

### Task 3: Service — `evaluate_recorder` with caching

**Files:**
- Modify: `backend/app/evaluation/service.py`
- Modify: `backend/app/evaluation/tests/test_service.py`

- [ ] **Step 1: Write the failing test for `evaluate_recorder`**

Append to `backend/app/evaluation/tests/test_service.py`:

```python
from app.evaluation.service import evaluate_recorder, list_recorders_with_summary


def test_evaluate_daily_cn_fresh_recorder(qlib_ready):
    summaries = list_recorders_with_summary()
    target = next((s for s in summaries if s.experiment == "daily_cn_fresh"), None)
    if target is None:
        pytest.skip("no daily_cn_fresh recorder")

    result = evaluate_recorder(target.recorder_id, top_k=30, cost_bps=10)
    assert result.recorder_id == target.recorder_id
    assert result.experiment == "daily_cn_fresh"
    assert result.sample_size > 100, "expected >100 (date,symbol) pairs after label join"
    # IC for a real model should be in [-0.1, 0.1] — sanity bound
    assert -0.1 < result.scorecard.ic_mean < 0.1
    # IR should be finite (could be negative if the model is bad)
    assert -10 < result.scorecard.ir < 10
    # Acceptance has 5 detail keys per spec
    assert set(result.acceptance.details.keys()) == {
        "ic_mean", "ir", "max_drawdown", "daily_turnover", "regimes_all_positive",
    }


def test_evaluate_recorder_is_cached(qlib_ready):
    summaries = list_recorders_with_summary()
    target = next((s for s in summaries if s.experiment == "daily_cn_fresh"), None)
    if target is None:
        pytest.skip("no daily_cn_fresh recorder")

    import time
    t0 = time.time()
    evaluate_recorder(target.recorder_id)
    t_first = time.time() - t0
    t0 = time.time()
    evaluate_recorder(target.recorder_id)
    t_cached = time.time() - t0
    # Cache hit should be at least 50x faster than the first call.
    assert t_cached * 50 < t_first, f"first={t_first:.2f}s cached={t_cached:.3f}s — caching not effective"


def test_force_refresh_bypasses_cache(qlib_ready):
    summaries = list_recorders_with_summary()
    target = next((s for s in summaries if s.experiment == "daily_cn_fresh"), None)
    if target is None:
        pytest.skip("no daily_cn_fresh recorder")
    r1 = evaluate_recorder(target.recorder_id)
    r2 = evaluate_recorder(target.recorder_id, force_refresh=True)
    # Same recorder, same data, same metrics — but computed_at differs.
    assert r1.scorecard.ic_mean == r2.scorecard.ic_mean
    assert r1.computed_at != r2.computed_at
```

- [ ] **Step 2: Run — expect failure**

```
cd E:/Projects/qlib/backend
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/evaluation/tests/test_service.py::test_evaluate_daily_cn_fresh_recorder -v
```

Expected: `ImportError: cannot import name 'evaluate_recorder'`.

- [ ] **Step 3: Implement `evaluate_recorder` and replace the cache stubs**

Replace the cache stubs (`_cache_has` and `_cache_quick_look`) at the bottom of `backend/app/evaluation/service.py` with the implementations below.

The cache strategy: `_evaluate_cached` is wrapped by `lru_cache` and does the heavy work. The public `evaluate_recorder` wrapper drives the cache and also updates `_CACHE_SEEN`/`_CACHE_RESULTS` sidecar maps that `_cache_has` and `_cache_quick_look` read from (so `list_recorders_with_summary` can show "已评估" + quick-look metrics in the table).

```python
import functools
from datetime import datetime, timezone

from qlib.workflow import R as _R
from qlib.data import D as _D

from app.evaluation.schemas import (
    AcceptanceResult,
    EvalResult,
    RegimeMetrics,
    ScorecardData,
)

# Spec §8 multi-regime segments. evaluate_recorder filters to those overlapping
# the recorder's prediction window; for recent-only recorders (e.g. 2025+) we
# always synthesize a "Recent" segment covering the full window.
_REGIME_SEGMENTS: list[tuple[str, str, str]] = [
    ("2018 Bear", "2018-01-01", "2018-12-31"),
    ("2019-20 Recovery", "2019-01-01", "2020-02-28"),
    ("2020-21 COVID liquidity", "2020-03-01", "2021-02-28"),
    ("2021-22 High vol", "2021-03-01", "2022-10-31"),
    ("2022-24 AI rally", "2022-11-01", "2024-12-31"),
]

# Sidecar maps populated by evaluate_recorder() so list view can show
# "已评估" status + quick-look IC/IR. Cleared on force_refresh.
_CACHE_SEEN: set[str] = set()
_CACHE_RESULTS: dict[str, EvalResult] = {}


def evaluate_recorder(
    recorder_id: str,
    top_k: int = 30,
    cost_bps: float = 10.0,
    force_refresh: bool = False,
) -> EvalResult:
    """Run the full 8-metric scorecard + regime split + acceptance check.

    Results cached in-process by (recorder_id, top_k, cost_bps).
    `force_refresh=True` clears the entire cache and recomputes.
    """
    if force_refresh:
        _evaluate_cached.cache_clear()
        _CACHE_SEEN.clear()
        _CACHE_RESULTS.clear()
    result = _evaluate_cached(recorder_id, top_k, cost_bps)
    _CACHE_SEEN.add(recorder_id)
    _CACHE_RESULTS[recorder_id] = result
    return result


@functools.lru_cache(maxsize=32)
def _evaluate_cached(recorder_id: str, top_k: int, cost_bps: float) -> EvalResult:
    """Heavy path: load pred.pkl, fetch labels, compute scorecard + regimes."""
    from production.metrics import compute_scorecard, regime_split
    from production.validate_acceptance import check_acceptance

    init_qlib_once()
    exp_name = _experiment_for_recorder(recorder_id)
    if exp_name is None:
        raise ValueError(f"recorder {recorder_id} not found in any experiment")

    pred = _load_pred_as_series(recorder_id, exp_name)
    if pred.empty:
        raise ValueError(f"recorder {recorder_id}: pred.pkl empty or missing")

    labels = _fetch_open_to_open_labels(pred)

    # Run scorecard
    scorecard_dict = compute_scorecard(pred, labels, top_k=top_k, bps=cost_bps)
    scorecard = ScorecardData(**scorecard_dict)

    # Regime split — only segments overlapping the prediction range, plus a Recent catch-all
    overlapping = _overlapping_regimes(pred)
    regimes_raw = regime_split(pred, labels, [(s, e) for _, s, e in overlapping])
    regimes: list[RegimeMetrics] = []
    for (label_name, start, end), key in zip(overlapping, regimes_raw.keys()):
        seg = regimes_raw[key]
        # Sample size = rows in the segment after label join
        mask = (
            (pred.index.get_level_values("datetime") >= pd.Timestamp(start))
            & (pred.index.get_level_values("datetime") <= pd.Timestamp(end))
        )
        regimes.append(
            RegimeMetrics(
                label=label_name,
                start=start,
                end=end,
                sample_size=int(mask.sum()),
                scorecard=ScorecardData(**seg),
            )
        )

    regime_irs = {r.label: r.scorecard.ir for r in regimes}
    acceptance_dict = check_acceptance(scorecard_dict, regime_irs)
    acceptance = AcceptanceResult(**acceptance_dict)

    # Compute the actual evaluation window from labels (post-join)
    joined_dates = (
        pd.concat([pred.rename("p"), labels.rename("y")], axis=1).dropna()
        .index.get_level_values("datetime")
    )
    window_start = str(joined_dates.min().date()) if len(joined_dates) else ""
    window_end = str(joined_dates.max().date()) if len(joined_dates) else ""

    rec = _R.get_recorder(recorder_id=recorder_id, experiment_name=exp_name)
    run_name = rec.info.get("name", recorder_id[:8])

    return EvalResult(
        recorder_id=recorder_id,
        experiment=exp_name,
        run_name=run_name,
        computed_at=datetime.now(timezone.utc).isoformat(),
        window_start=window_start,
        window_end=window_end,
        sample_size=len(joined_dates),
        top_k=top_k,
        cost_bps=cost_bps,
        scorecard=scorecard,
        regimes=regimes,
        acceptance=acceptance,
    )


def _experiment_for_recorder(recorder_id: str) -> str | None:
    """Find which experiment owns this recorder_id."""
    for exp_id in _list_experiment_ids():
        rec_dir = Settings().mlruns_path / exp_id / recorder_id
        if rec_dir.is_dir():
            return _experiment_name(exp_id)
    return None


def _load_pred_as_series(recorder_id: str, exp_name: str) -> pd.Series:
    """Load pred.pkl (or pred_5d.pkl etc.) as a 1-col Series indexed by (datetime, instrument).
    For multi-column DataFrames (ensemble output), uses the 'score' column."""
    exp_id = _find_exp_id_for_name(exp_name)
    if exp_id is None:
        return pd.Series(dtype="float64")
    artifacts = Settings().mlruns_path / exp_id / recorder_id / "artifacts"
    candidates = [artifacts / "pred.pkl", artifacts / "pred_5d.pkl",
                  artifacts / "pred_1d.pkl", artifacts / "pred_20d.pkl"]
    for c in candidates:
        if not c.exists():
            continue
        df = pd.read_pickle(c)
        if isinstance(df, pd.Series):
            return _ensure_index(df.rename("score"))
        if "score" in df.columns:
            return _ensure_index(df["score"])
        return _ensure_index(df.iloc[:, 0])
    return pd.Series(dtype="float64")


def _ensure_index(s: pd.Series) -> pd.Series:
    if s.index.names != ["datetime", "instrument"]:
        s.index = s.index.set_names(["datetime", "instrument"])
    return s.sort_index()


def _fetch_open_to_open_labels(pred: pd.Series) -> pd.Series:
    """Pull Ref($open, -2) / Ref($open, -1) - 1 from qlib for the same
    (date, symbol) range as pred. Returns a Series with the same index layout."""
    symbols = sorted(pred.index.get_level_values("instrument").unique().tolist())
    dates = pred.index.get_level_values("datetime")
    start = (pd.Timestamp(dates.min()) - pd.Timedelta(days=5)).date()
    end = (pd.Timestamp(dates.max()) + pd.Timedelta(days=10)).date()
    df = _D.features(
        instruments=symbols,
        fields=["Ref($open, -2) / Ref($open, -1) - 1"],
        start_time=str(start),
        end_time=str(end),
    )
    df.columns = ["y"]
    s = df["y"]
    if s.index.names != ["datetime", "instrument"]:
        # qlib usually returns (instrument, datetime); normalize.
        s.index.names = ["instrument", "datetime"]
        s = s.swaplevel().sort_index()
    return s


def _overlapping_regimes(pred: pd.Series) -> list[tuple[str, str, str]]:
    """Return the spec regime segments whose [start, end] overlap the
    prediction window. Always appends a 'Recent' synthetic segment covering
    the full prediction window so recorders with only 2025+ predictions still
    get something."""
    dates = pred.index.get_level_values("datetime")
    pred_start = pd.Timestamp(dates.min())
    pred_end = pd.Timestamp(dates.max())
    out: list[tuple[str, str, str]] = []
    for label, start_s, end_s in _REGIME_SEGMENTS:
        s = pd.Timestamp(start_s)
        e = pd.Timestamp(end_s)
        if e < pred_start or s > pred_end:
            continue
        out.append((label, start_s, end_s))
    # Catch-all "Recent" segment over the whole prediction range
    out.append(("Recent (full window)", str(pred_start.date()), str(pred_end.date())))
    return out


def _cache_has(recorder_id: str) -> bool:
    """True iff evaluate_recorder has been called for this recorder
    since process startup (or last force_refresh)."""
    return recorder_id in _CACHE_SEEN


def _cache_quick_look(recorder_id: str) -> tuple[float | None, float | None, bool | None]:
    """If cached, return (ic_mean, ir, acceptance_passed) for the list view's quick column."""
    res = _CACHE_RESULTS.get(recorder_id)
    if res is None:
        return (None, None, None)
    return (res.scorecard.ic_mean, res.scorecard.ir, res.acceptance.passed)
```

- [ ] **Step 4: Run tests — expect PASS**

```
cd E:/Projects/qlib/backend
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/evaluation/tests/test_service.py -v
```

Expected: 5 tests PASS (2 from T2, 3 new from T3). The first eval call may take 30-90s; cached call < 100ms.

- [ ] **Step 5: Commit**

```
cd E:/Projects/qlib
git add backend/app/evaluation/service.py backend/app/evaluation/tests/test_service.py
git commit -m "feat(evaluation): evaluate_recorder with lru_cache + regime overlap detection"
```

---

### Task 4: Service — `compare_recorders` with paired t-test

**Files:**
- Modify: `backend/app/evaluation/service.py`
- Modify: `backend/app/evaluation/tests/test_service.py`

- [ ] **Step 1: Write the failing test**

Append to `backend/app/evaluation/tests/test_service.py`:

```python
from app.evaluation.service import compare_recorders


def test_compare_recorder_with_itself_yields_no_significance(qlib_ready):
    summaries = list_recorders_with_summary()
    target = next((s for s in summaries if s.experiment == "daily_cn_fresh"), None)
    if target is None:
        pytest.skip("no daily_cn_fresh recorder")

    cmp = compare_recorders(target.recorder_id, target.recorder_id)
    # A recorder vs itself: identical scorecards, p-value ~ 1 (or NaN per ttest_rel on identical)
    assert cmp.ic_delta == pytest.approx(0.0, abs=1e-9)
    assert cmp.ir_delta == pytest.approx(0.0, abs=1e-9)
    assert cmp.significant_at_05 is False
```

- [ ] **Step 2: Run — expect failure**

```
cd E:/Projects/qlib/backend
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/evaluation/tests/test_service.py::test_compare_recorder_with_itself_yields_no_significance -v
```

Expected: ImportError on `compare_recorders`.

- [ ] **Step 3: Implement `compare_recorders`**

Append to `backend/app/evaluation/service.py`:

```python
def compare_recorders(recorder_a_id: str, recorder_b_id: str, top_k: int = 30, cost_bps: float = 10.0) -> "CompareResult":
    """Compare two recorders side-by-side. Runs evaluate_recorder for each
    (uses cache), aligns their daily-IC series, and runs a paired t-test
    on the differences."""
    from app.evaluation.schemas import CompareResult
    from production.metrics import paired_ttest, _daily_ic

    a = evaluate_recorder(recorder_a_id, top_k=top_k, cost_bps=cost_bps)
    b = evaluate_recorder(recorder_b_id, top_k=top_k, cost_bps=cost_bps)

    # Recompute daily IC series for each to feed paired_ttest. We pay the
    # label-fetch cost twice but it's the same data so it's fast on the 2nd call.
    pred_a = _load_pred_as_series(recorder_a_id, a.experiment)
    pred_b = _load_pred_as_series(recorder_b_id, b.experiment)
    labels_a = _fetch_open_to_open_labels(pred_a)
    labels_b = _fetch_open_to_open_labels(pred_b)
    ic_a = _daily_ic(pred_a, labels_a)
    ic_b = _daily_ic(pred_b, labels_b)

    # Handle identical recorders cleanly (ttest_rel of identical series is NaN)
    if recorder_a_id == recorder_b_id:
        t_stat, p_value = 0.0, 1.0
    else:
        try:
            t_stat, p_value = paired_ttest(ic_b, ic_a)
        except Exception:
            t_stat, p_value = float("nan"), float("nan")

    ic_delta = b.scorecard.ic_mean - a.scorecard.ic_mean
    ir_delta = b.scorecard.ir - a.scorecard.ir

    if not (p_value < 0.05):
        verdict = "no significant difference"
    elif ic_delta > 0:
        verdict = "b significantly better"
    else:
        verdict = "a significantly better"

    return CompareResult(
        a=a,
        b=b,
        paired_t_stat=float(t_stat) if t_stat == t_stat else 0.0,  # NaN -> 0
        paired_p_value=float(p_value) if p_value == p_value else 1.0,
        significant_at_05=bool(p_value < 0.05) if p_value == p_value else False,
        ic_delta=ic_delta,
        ir_delta=ir_delta,
        verdict=verdict,
    )
```

- [ ] **Step 4: Run tests — expect PASS**

```
cd E:/Projects/qlib/backend
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/evaluation/tests/test_service.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```
cd E:/Projects/qlib
git add backend/app/evaluation/service.py backend/app/evaluation/tests/test_service.py
git commit -m "feat(evaluation): compare_recorders with paired t-test"
```

---

### Task 5: Router with 4 endpoints + integration tests

**Files:**
- Create: `backend/app/evaluation/router.py`
- Create: `backend/app/evaluation/tests/test_router.py`
- Modify: `backend/app/main.py` (wire router)

- [ ] **Step 1: Write the failing router test**

`backend/app/evaluation/tests/test_router.py`:

```python
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app


@pytest.fixture
async def client():
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        async with app.router.lifespan_context(app):
            yield c


async def test_list_recorders(client):
    r = await client.get("/api/evaluation/recorders")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    if not body:
        pytest.skip("no recorders in mlruns")
    s = body[0]
    assert "recorder_id" in s and "experiment" in s and "run_name" in s


async def test_run_evaluation(client):
    listing = (await client.get("/api/evaluation/recorders")).json()
    target = next((s for s in listing if s["experiment"] == "daily_cn_fresh"), None)
    if target is None:
        pytest.skip("no daily_cn_fresh recorder")
    r = await client.post("/api/evaluation/run", json={"recorder_id": target["recorder_id"]})
    assert r.status_code == 200
    body = r.json()
    assert body["recorder_id"] == target["recorder_id"]
    assert "scorecard" in body and "ic_mean" in body["scorecard"]
    assert "regimes" in body and isinstance(body["regimes"], list)
    assert "acceptance" in body and "passed" in body["acceptance"]


async def test_get_cached_results(client):
    listing = (await client.get("/api/evaluation/recorders")).json()
    target = next((s for s in listing if s["experiment"] == "daily_cn_fresh"), None)
    if target is None:
        pytest.skip("no daily_cn_fresh recorder")
    # Warm cache
    await client.post("/api/evaluation/run", json={"recorder_id": target["recorder_id"]})
    # Read from cache
    r = await client.get(f"/api/evaluation/results/{target['recorder_id']}")
    assert r.status_code == 200
    assert r.json()["recorder_id"] == target["recorder_id"]


async def test_get_missing_recorder_returns_404(client):
    r = await client.get("/api/evaluation/results/nonexistent_recorder_id_xyz")
    assert r.status_code == 404


async def test_compare(client):
    listing = (await client.get("/api/evaluation/recorders")).json()
    target = next((s for s in listing if s["experiment"] == "daily_cn_fresh"), None)
    if target is None:
        pytest.skip("no daily_cn_fresh recorder")
    r = await client.get(
        f"/api/evaluation/compare?a={target['recorder_id']}&b={target['recorder_id']}"
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ic_delta"] == 0.0
    assert body["verdict"] == "no significant difference"
```

- [ ] **Step 2: Run — expect failure (router not wired yet)**

```
cd E:/Projects/qlib/backend
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/evaluation/tests/test_router.py -v
```

Expected: 404 on every endpoint (router not in app yet).

- [ ] **Step 3: Implement the router**

`backend/app/evaluation/router.py`:

```python
from fastapi import APIRouter, HTTPException, Query

from app.evaluation import service
from app.evaluation.schemas import (
    CompareResult,
    EvalResult,
    EvalRunRequest,
    RecorderSummary,
)

router = APIRouter()


@router.get("/recorders", response_model=list[RecorderSummary])
def list_recorders():
    """Return a summary row per recorder across all experiments. Cheap."""
    return service.list_recorders_with_summary()


@router.post("/run", response_model=EvalResult)
def run_eval(payload: EvalRunRequest):
    """Trigger evaluation for a recorder. Returns the full scorecard +
    regime breakdown + acceptance result. Cached in-process."""
    try:
        return service.evaluate_recorder(
            recorder_id=payload.recorder_id,
            top_k=payload.top_k,
            cost_bps=payload.cost_bps,
            force_refresh=payload.force_refresh,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/results/{recorder_id}", response_model=EvalResult)
def get_results(recorder_id: str):
    """Return cached evaluation result. 404 if not yet computed."""
    if recorder_id not in service._CACHE_SEEN:
        raise HTTPException(status_code=404, detail=f"no cached evaluation for {recorder_id}")
    return service._CACHE_RESULTS[recorder_id]


@router.get("/compare", response_model=CompareResult)
def compare(a: str = Query(...), b: str = Query(...), top_k: int = 30, cost_bps: float = 10.0):
    """Compare two recorders side-by-side with paired t-test on daily IC."""
    try:
        return service.compare_recorders(a, b, top_k=top_k, cost_bps=cost_bps)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
```

- [ ] **Step 4: Wire the router in `backend/app/main.py`**

Locate the existing `app.include_router(...)` calls in `backend/app/main.py` and add this line in the same cluster:

```python
    from app.evaluation.router import router as evaluation_router
    app.include_router(evaluation_router, prefix="/api/evaluation", tags=["evaluation"])
```

(The import can also live at module top; place it consistently with other routers.)

- [ ] **Step 5: Run tests — expect PASS**

```
cd E:/Projects/qlib/backend
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/evaluation/tests/test_router.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 6: Smoke-test through the running backend (restart it if needed)**

```
F:/Tools/Anaconda/envs/qlib/python.exe -c "
import urllib.request, json
recs = json.loads(urllib.request.urlopen('http://127.0.0.1:8000/api/evaluation/recorders', timeout=30).read())
print(f'{len(recs)} recorders: {[(r[\"experiment\"], r[\"recorder_id\"][:8]) for r in recs]}')
"
```

- [ ] **Step 7: Commit**

```
cd E:/Projects/qlib
git add backend/app/evaluation/router.py backend/app/evaluation/tests/test_router.py backend/app/main.py
git commit -m "feat(evaluation): 4 REST endpoints (/recorders, /run, /results, /compare)"
```

---

## Phase B — CLI

### Task 6: `production/eval_recorder.py` CLI

**Files:**
- Create: `production/eval_recorder.py`
- Create: `production/reports/.gitkeep`

- [ ] **Step 1: Create the reports directory placeholder**

```
cd E:/Projects/qlib
mkdir -p production/reports
echo "" > production/reports/.gitkeep
```

- [ ] **Step 2: Implement the CLI**

`production/eval_recorder.py`:

```python
"""Evaluate a qlib recorder against the spec §8 8-metric scorecard.

Usage:
  python -m production.eval_recorder <recorder_id> [--top-k 30] [--bps 10] [--out production/reports]
  python -m production.eval_recorder --list                # list all recorders
  python -m production.eval_recorder --compare A B         # paired t-test between two recorders
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

# --- sys.path fix mirroring rolling_train.py: put site-packages first so the
# conda-installed qlib wins over the in-repo source.
import sys as _sys
import sysconfig as _sysconfig
_PURELIB = _sysconfig.get_paths().get("purelib")
if _PURELIB and _PURELIB not in _sys.path[:1]:
    _sys.path.insert(0, _PURELIB)

# Add backend/ so the backend's evaluation service is importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "backend"))

warnings.filterwarnings("ignore", category=FutureWarning)


def _render_markdown(result) -> str:
    s = result.scorecard
    a = result.acceptance
    lines = []
    lines.append(f"# Evaluation: {result.run_name} ({result.experiment})")
    lines.append("")
    lines.append(f"- **recorder_id**: `{result.recorder_id}`")
    lines.append(f"- **computed_at**: {result.computed_at}")
    lines.append(f"- **window**: {result.window_start} → {result.window_end}")
    lines.append(f"- **samples**: {result.sample_size:,} (date,symbol) pairs")
    lines.append(f"- **portfolio**: TopK={result.top_k}, cost={result.cost_bps} bps")
    lines.append("")
    lines.append("## Scorecard")
    lines.append("")
    lines.append("| Metric | Value | Threshold (spec §11) | Pass? |")
    lines.append("|---|---|---|---|")
    lines.append(f"| IC mean | {s.ic_mean:+.4f} | ≥ 0.030 | {'✅' if a.details.get('ic_mean') else '❌'} |")
    lines.append(f"| RIC mean | {s.ric_mean:+.4f} | — | — |")
    lines.append(f"| ICIR | {s.icir:+.3f} | ≥ 0.40 | {'✅' if s.icir >= 0.40 else '❌'} |")
    lines.append(f"| Top-bottom monthly | {s.top_bottom_spread_monthly:+.2f}% | ≥ 1.5% | {'✅' if s.top_bottom_spread_monthly >= 1.5 else '❌'} |")
    lines.append(f"| Annual return | {s.annual_excess_return*100:+.2f}% | ≥ +15% | {'✅' if s.annual_excess_return >= 0.15 else '❌'} |")
    lines.append(f"| IR (cost-adj) | {s.ir:+.3f} | ≥ 2.5 | {'✅' if a.details.get('ir') else '❌'} |")
    lines.append(f"| Max drawdown | {s.max_drawdown*100:+.2f}% | ≥ -15% | {'✅' if a.details.get('max_drawdown') else '❌'} |")
    lines.append(f"| Daily turnover | {s.daily_turnover*100:.2f}% | ≤ 20% | {'✅' if a.details.get('daily_turnover') else '❌'} |")
    lines.append("")
    lines.append("## Regime breakdown")
    lines.append("")
    lines.append("| Period | Samples | IC mean | IR | MDD |")
    lines.append("|---|---|---|---|---|")
    for r in result.regimes:
        rs = r.scorecard
        lines.append(f"| {r.label} | {r.sample_size:,} | {rs.ic_mean:+.4f} | {rs.ir:+.3f} | {rs.max_drawdown*100:+.2f}% |")
    lines.append("")
    lines.append(f"## Acceptance: **{'PASS ✅' if a.passed else 'FAIL ❌'}**")
    lines.append("")
    for k, v in a.details.items():
        lines.append(f"- {k}: {'✅' if v else '❌'}")
    return "\n".join(lines)


def _cmd_list():
    from app.evaluation.service import list_recorders_with_summary
    recs = list_recorders_with_summary()
    for r in recs:
        rng = f"{r.pred_start}..{r.pred_end}" if r.pred_start else "no preds"
        print(f"  {r.recorder_id[:12]}  exp={r.experiment:<24}  run={r.run_name:<28}  range={rng}  rows={r.pred_rows}")


def _cmd_eval(args):
    from app.evaluation.service import evaluate_recorder
    print(f"Evaluating {args.recorder_id} (top_k={args.top_k}, bps={args.bps})...", flush=True)
    result = evaluate_recorder(args.recorder_id, top_k=args.top_k, cost_bps=args.bps)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"{args.recorder_id[:12]}_{ts}.json"
    md_path = out_dir / f"{args.recorder_id[:12]}_{ts}.md"
    json_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(result), encoding="utf-8")
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"\nVerdict: {'PASS' if result.acceptance.passed else 'FAIL'}")


def _cmd_compare(args):
    from app.evaluation.service import compare_recorders
    cmp = compare_recorders(args.a, args.b)
    print(f"A: {cmp.a.run_name} ({cmp.a.experiment})  IC={cmp.a.scorecard.ic_mean:+.4f}  IR={cmp.a.scorecard.ir:+.3f}")
    print(f"B: {cmp.b.run_name} ({cmp.b.experiment})  IC={cmp.b.scorecard.ic_mean:+.4f}  IR={cmp.b.scorecard.ir:+.3f}")
    print(f"")
    print(f"  IC delta: {cmp.ic_delta:+.4f}")
    print(f"  IR delta: {cmp.ir_delta:+.3f}")
    print(f"  t-stat:   {cmp.paired_t_stat:+.3f}")
    print(f"  p-value:  {cmp.paired_p_value:.4f}")
    print(f"  Verdict:  {cmp.verdict}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a qlib recorder.")
    parser.add_argument("recorder_id", nargs="?", help="recorder UUID to evaluate")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--bps", type=float, default=10.0)
    parser.add_argument("--out", default="production/reports")
    parser.add_argument("--list", action="store_true", help="list all available recorders")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"),
                        help="paired t-test between two recorders")
    args = parser.parse_args()

    if args.list:
        _cmd_list()
        return 0
    if args.compare:
        args.a, args.b = args.compare
        _cmd_compare(args)
        return 0
    if not args.recorder_id:
        parser.print_help()
        return 1
    _cmd_eval(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Smoke-test the list command**

```
cd E:/Projects/qlib
F:/Tools/Anaconda/envs/qlib/python.exe -m production.eval_recorder --list
```

Expected: prints rows for each recorder (daily_cn_fresh, 3× rolling_v2_ensemble sub-recorders).

- [ ] **Step 4: Smoke-test the eval command on daily_cn_fresh**

Pick the daily_cn_fresh recorder ID from the listing and run:

```
F:/Tools/Anaconda/envs/qlib/python.exe -m production.eval_recorder f29f042f72634226aa0dc7782d4873d9
```

Expected: ~30-90s run; prints "Wrote production/reports/f29f042f_<ts>.json" + ".md"; then "Verdict: FAIL" (expected for daily_cn_fresh since IC ~ 0.024).

- [ ] **Step 5: Commit**

```
cd E:/Projects/qlib
git add production/eval_recorder.py production/reports/.gitkeep
git commit -m "feat(eval): production/eval_recorder.py CLI (list/eval/compare)"
```

---

## Phase C — Frontend

### Task 7: API types + hooks + nav

**Files:**
- Modify: `frontend/src/api/types.gen.ts` (regen)
- Modify: `frontend/src/api/client.ts` (add `evaluation` namespace)
- Create: `frontend/src/pages/evaluation/types.ts`
- Create: `frontend/src/pages/evaluation/hooks.ts`
- Modify: `frontend/src/components/Layout.tsx` (add nav link)
- Modify: `frontend/src/App.tsx` (add route)

- [ ] **Step 1: Regen types**

```
cd E:/Projects/qlib/frontend
npm run gen:api
```

Verify:
```
grep -n "RecorderSummary\|EvalResult\|CompareResult\|evaluation" frontend/src/api/types.gen.ts | head -10
```

- [ ] **Step 2: Add `evaluation` to `frontend/src/api/client.ts`**

Inside the `api = {...}` object, add a sibling to `models:`:

```typescript
  evaluation: {
    recorders: () => {
      type R = paths['/api/evaluation/recorders']['get']['responses']['200']['content']['application/json'];
      return request<R>('/api/evaluation/recorders');
    },
    run: (body: { recorder_id: string; top_k?: number; cost_bps?: number; force_refresh?: boolean }) => {
      type R = paths['/api/evaluation/run']['post']['responses']['200']['content']['application/json'];
      return request<R>('/api/evaluation/run', {
        method: 'POST',
        body: JSON.stringify(body),
      });
    },
    results: (recorder_id: string) => {
      type R = paths['/api/evaluation/results/{recorder_id}']['get']['responses']['200']['content']['application/json'];
      return request<R>(`/api/evaluation/results/${encodeURIComponent(recorder_id)}`);
    },
    compare: (a: string, b: string) => {
      type R = paths['/api/evaluation/compare']['get']['responses']['200']['content']['application/json'];
      const q = new URLSearchParams({ a, b });
      return request<R>(`/api/evaluation/compare?${q.toString()}`);
    },
  },
```

- [ ] **Step 3: Create local types module**

`frontend/src/pages/evaluation/types.ts`:

```typescript
import type { components } from '@/api/types.gen';

export type RecorderSummary = components['schemas']['RecorderSummary'];
export type EvalResult = components['schemas']['EvalResult'];
export type CompareResult = components['schemas']['CompareResult'];
export type ScorecardData = components['schemas']['ScorecardData'];
export type RegimeMetrics = components['schemas']['RegimeMetrics'];
export type AcceptanceResult = components['schemas']['AcceptanceResult'];

/** Spec §11 thresholds. Mirror of backend production/validate_acceptance.THRESHOLDS. */
export const THRESHOLDS = {
  ic_mean: 0.030,
  ir: 2.5,
  max_drawdown: -0.15,
  daily_turnover: 0.20,
  icir: 0.40,                          // not in backend THRESHOLDS but visible in UI per spec
  top_bottom_spread_monthly: 1.5,
  annual_excess_return: 0.15,
} as const;

/** Color a metric value vs. its threshold. Returns 'green' for pass, 'red' for fail, 'gray' for N/A. */
export function passColor(metric: keyof typeof THRESHOLDS, value: number | null | undefined): 'green' | 'red' | 'gray' {
  if (value === null || value === undefined || Number.isNaN(value)) return 'gray';
  const t = THRESHOLDS[metric];
  // max_drawdown threshold is negative (-0.15 means "must be ≥ -0.15")
  if (metric === 'max_drawdown') {
    return value >= t ? 'green' : 'red';
  }
  // daily_turnover is "must be ≤ threshold"
  if (metric === 'daily_turnover') {
    return value <= t ? 'green' : 'red';
  }
  // All other metrics: "must be ≥ threshold"
  return value >= t ? 'green' : 'red';
}
```

- [ ] **Step 4: Create the hooks**

`frontend/src/pages/evaluation/hooks.ts`:

```typescript
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { api } from '@/api/client';

export function useRecorders() {
  return useQuery({
    queryKey: ['evaluation', 'recorders'],
    queryFn: () => api.evaluation.recorders(),
    staleTime: 60_000,
  });
}

export function useEvaluation(recorder_id: string | null) {
  return useQuery({
    queryKey: ['evaluation', 'result', recorder_id],
    queryFn: () => api.evaluation.results(recorder_id!),
    enabled: !!recorder_id,
    staleTime: Infinity,
    retry: false,  // 404 = "not computed yet"; don't retry
  });
}

export function useRunEvaluation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { recorder_id: string; force_refresh?: boolean }) =>
      api.evaluation.run(body),
    onSuccess: (data) => {
      qc.setQueryData(['evaluation', 'result', data.recorder_id], data);
      qc.invalidateQueries({ queryKey: ['evaluation', 'recorders'] });
    },
  });
}

export function useCompare(a: string | null, b: string | null) {
  return useQuery({
    queryKey: ['evaluation', 'compare', a, b],
    queryFn: () => api.evaluation.compare(a!, b!),
    enabled: !!a && !!b,
    staleTime: Infinity,
  });
}
```

- [ ] **Step 5: Add nav link + route**

In `frontend/src/components/Layout.tsx`, find the existing `<NavLink>` block (with Settings / Picks / Portfolio / Dashboard) and add:

```tsx
<NavLink to="/evaluation" className={navCls}>Evaluation</NavLink>
```

(Match the existing `navCls` pattern.)

In `frontend/src/App.tsx`, find the `<Routes>` block and add:

```tsx
import Evaluation from './pages/Evaluation';
// ... inside <Routes>:
<Route path="/evaluation" element={<Evaluation />} />
```

- [ ] **Step 6: Create a stub `Evaluation.tsx` so the route compiles (will be filled in T8)**

`frontend/src/pages/Evaluation.tsx`:

```tsx
import { useRecorders } from './evaluation/hooks';

export default function Evaluation() {
  const { data, isPending, error } = useRecorders();
  if (isPending) return <div className="p-6 text-[#8b949e]">Loading recorders…</div>;
  if (error) return <div className="p-6 text-red-400">Error: {String(error)}</div>;
  return (
    <div className="p-6">
      <h1 className="text-2xl font-semibold mb-4">Model Evaluation</h1>
      <p className="text-sm text-[#8b949e] mb-4">
        Stub — list view + detail + compare are added in Task 8/9/10.
      </p>
      <pre className="text-xs">{JSON.stringify(data, null, 2)}</pre>
    </div>
  );
}
```

- [ ] **Step 7: Typecheck**

```
cd E:/Projects/qlib/frontend
npx tsc --noEmit
```

Expected: 0 errors.

- [ ] **Step 8: Commit**

```
cd E:/Projects/qlib
git add frontend/src/api/types.gen.ts frontend/src/api/client.ts frontend/src/pages/evaluation/types.ts frontend/src/pages/evaluation/hooks.ts frontend/src/components/Layout.tsx frontend/src/App.tsx frontend/src/pages/Evaluation.tsx
git commit -m "feat(evaluation): API types + hooks + nav + route stub"
```

---

### Task 8: List view — table of all recorders

**Files:**
- Modify: `frontend/src/pages/Evaluation.tsx`
- Create: `frontend/src/pages/evaluation/RecorderRow.tsx`

- [ ] **Step 1: Create `RecorderRow.tsx`**

`frontend/src/pages/evaluation/RecorderRow.tsx`:

```tsx
import { cn } from '@/lib/utils';
import type { RecorderSummary } from './types';
import { passColor } from './types';

interface Props {
  rec: RecorderSummary;
  selected: boolean;
  onSelect: () => void;
  onEvaluate: () => void;
  evaluating: boolean;
}

export function RecorderRow({ rec, selected, onSelect, onEvaluate, evaluating }: Props) {
  const icColor = rec.ic_mean != null ? passColor('ic_mean', rec.ic_mean) : 'gray';
  const irColor = rec.ir != null ? passColor('ir', rec.ir) : 'gray';

  return (
    <tr
      className={cn(
        'border-b border-[#21262d] hover:bg-[#161b22] transition cursor-pointer',
        selected && 'bg-[#1f6feb22]',
      )}
      onClick={onSelect}
    >
      <td className="py-2 pr-4">
        <div className="font-mono text-xs text-[#58a6ff]">{rec.recorder_id.slice(0, 12)}</div>
        <div className="text-xs text-[#8b949e]">{rec.run_name}</div>
      </td>
      <td className="py-2 pr-4 text-sm">{rec.experiment}</td>
      <td className="py-2 pr-4 text-xs text-[#8b949e]">
        {rec.pred_start && rec.pred_end ? `${rec.pred_start} → ${rec.pred_end}` : '—'}
        <div>{rec.pred_rows?.toLocaleString() ?? '—'} rows</div>
      </td>
      <td className={cn('py-2 pr-4 text-right font-mono', colorClass(icColor))}>
        {rec.ic_mean != null ? rec.ic_mean.toFixed(4) : <em className="text-[#6e7681]">not run</em>}
      </td>
      <td className={cn('py-2 pr-4 text-right font-mono', colorClass(irColor))}>
        {rec.ir != null ? rec.ir.toFixed(3) : '—'}
      </td>
      <td className="py-2 pr-4">
        {rec.acceptance_passed === true ? (
          <span className="text-green-400">✅ PASS</span>
        ) : rec.acceptance_passed === false ? (
          <span className="text-red-400">❌ FAIL</span>
        ) : (
          <span className="text-[#6e7681]">—</span>
        )}
      </td>
      <td className="py-2 pr-4">
        <button
          className="text-xs px-2 py-1 rounded bg-[#21262d] hover:bg-[#30363d] border border-[#30363d] disabled:opacity-50"
          disabled={evaluating}
          onClick={(e) => {
            e.stopPropagation();
            onEvaluate();
          }}
        >
          {evaluating ? '运行中…' : rec.has_eval ? '重跑' : '评估'}
        </button>
      </td>
    </tr>
  );
}

function colorClass(c: 'green' | 'red' | 'gray'): string {
  switch (c) {
    case 'green': return 'text-green-400';
    case 'red': return 'text-red-400';
    case 'gray': return 'text-[#8b949e]';
  }
}
```

- [ ] **Step 2: Rewrite `Evaluation.tsx` with the list table**

```tsx
import { useState } from 'react';

import { useRecorders, useRunEvaluation } from './evaluation/hooks';
import { RecorderRow } from './evaluation/RecorderRow';

export default function Evaluation() {
  const { data: recorders, isPending, error } = useRecorders();
  const [selected, setSelected] = useState<string | null>(null);
  const runMut = useRunEvaluation();

  return (
    <div className="p-6 space-y-6 max-w-7xl">
      <header>
        <h1 className="text-2xl font-semibold">Model Evaluation</h1>
        <p className="text-sm text-[#8b949e] mt-1">
          按照 spec §8 评估每个 recorder 的 8 项指标 · 跑一次缓存进程 · 点击展开详情
        </p>
      </header>

      {error && <div className="text-red-400 text-sm">加载失败: {String(error)}</div>}
      {isPending && <div className="text-[#8b949e] text-sm">读取 recorder 列表…</div>}

      {recorders && (
        <div className="rounded-lg border border-[#30363d] bg-[#0d1117] p-5">
          <h2 className="text-sm font-semibold text-[#8b949e] uppercase tracking-wider mb-3">
            Recorders ({recorders.length})
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs uppercase tracking-wider text-[#6e7681] border-b border-[#30363d]">
                  <th className="py-2 pr-4">recorder / run</th>
                  <th className="py-2 pr-4">experiment</th>
                  <th className="py-2 pr-4">prediction window</th>
                  <th className="py-2 pr-4 text-right">IC mean</th>
                  <th className="py-2 pr-4 text-right">IR</th>
                  <th className="py-2 pr-4">acceptance</th>
                  <th className="py-2 pr-4"></th>
                </tr>
              </thead>
              <tbody>
                {recorders.map((rec) => (
                  <RecorderRow
                    key={rec.recorder_id}
                    rec={rec}
                    selected={selected === rec.recorder_id}
                    onSelect={() => setSelected(rec.recorder_id)}
                    onEvaluate={() => runMut.mutate({ recorder_id: rec.recorder_id })}
                    evaluating={runMut.isPending && runMut.variables?.recorder_id === rec.recorder_id}
                  />
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {selected && (
        <div className="rounded-lg border border-[#30363d] bg-[#0d1117] p-5">
          <h2 className="text-sm font-semibold text-[#8b949e] uppercase tracking-wider mb-3">
            Detail (T9 fills this in)
          </h2>
          <div className="text-xs font-mono text-[#8b949e]">selected: {selected}</div>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 3: Typecheck**

```
cd E:/Projects/qlib/frontend
npx tsc --noEmit
```

- [ ] **Step 4: Smoke test in browser**

Open `http://127.0.0.1:5173/evaluation`. Should show a table with all recorders. Click "评估" on the daily_cn_fresh row — wait ~30-90s — the row's IC/IR cells should populate.

- [ ] **Step 5: Commit**

```
cd E:/Projects/qlib
git add frontend/src/pages/Evaluation.tsx frontend/src/pages/evaluation/RecorderRow.tsx
git commit -m "feat(evaluation): list view with per-recorder evaluate button"
```

---

### Task 9: Detail view — scorecard + regime + acceptance

**Files:**
- Create: `frontend/src/pages/evaluation/ScorecardCard.tsx`
- Create: `frontend/src/pages/evaluation/RegimeChart.tsx`
- Create: `frontend/src/pages/evaluation/AcceptanceLights.tsx`
- Modify: `frontend/src/pages/Evaluation.tsx`

- [ ] **Step 1: Create `ScorecardCard.tsx`**

```tsx
import { cn } from '@/lib/utils';
import type { ScorecardData } from './types';
import { passColor } from './types';

interface Props {
  scorecard: ScorecardData;
  title?: string;
}

export function ScorecardCard({ scorecard: s, title = 'Scorecard' }: Props) {
  return (
    <div className="rounded-lg border border-[#30363d] bg-[#0d1117] p-5">
      <h3 className="text-sm font-semibold text-[#8b949e] uppercase tracking-wider mb-3">{title}</h3>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Cell label="IC mean" value={s.ic_mean.toFixed(4)} threshold="≥ 0.030" color={passColor('ic_mean', s.ic_mean)} />
        <Cell label="RIC mean" value={s.ric_mean.toFixed(4)} threshold="—" color="gray" />
        <Cell label="ICIR" value={s.icir.toFixed(3)} threshold="≥ 0.40" color={passColor('icir', s.icir)} />
        <Cell label="月度 top-bottom" value={s.top_bottom_spread_monthly.toFixed(2) + '%'} threshold="≥ 1.5%" color={passColor('top_bottom_spread_monthly', s.top_bottom_spread_monthly)} />
        <Cell label="年化收益" value={(s.annual_excess_return * 100).toFixed(2) + '%'} threshold="≥ +15%" color={passColor('annual_excess_return', s.annual_excess_return)} />
        <Cell label="IR" value={s.ir.toFixed(3)} threshold="≥ 2.5" color={passColor('ir', s.ir)} />
        <Cell label="最大回撤" value={(s.max_drawdown * 100).toFixed(2) + '%'} threshold="≥ -15%" color={passColor('max_drawdown', s.max_drawdown)} />
        <Cell label="日换手率" value={(s.daily_turnover * 100).toFixed(1) + '%'} threshold="≤ 20%" color={passColor('daily_turnover', s.daily_turnover)} />
      </div>
    </div>
  );
}

function Cell({ label, value, threshold, color }: { label: string; value: string; threshold: string; color: 'green' | 'red' | 'gray' }) {
  return (
    <div className="border border-[#30363d] rounded-md px-3 py-2">
      <div className="text-[10px] uppercase tracking-wider text-[#6e7681]">{label}</div>
      <div className={cn('text-lg font-mono mt-1', colorClass(color))}>{value}</div>
      <div className="text-[10px] text-[#6e7681] mt-1">阈值 {threshold}</div>
    </div>
  );
}

function colorClass(c: 'green' | 'red' | 'gray'): string {
  switch (c) {
    case 'green': return 'text-green-400';
    case 'red': return 'text-red-400';
    case 'gray': return 'text-[#e6edf3]';
  }
}
```

- [ ] **Step 2: Create `RegimeChart.tsx`**

```tsx
import type { RegimeMetrics } from './types';

interface Props {
  regimes: RegimeMetrics[];
}

export function RegimeChart({ regimes }: Props) {
  // Find the absolute max |IR| for scaling
  const maxAbs = Math.max(0.5, ...regimes.map((r) => Math.abs(r.scorecard.ir)));

  return (
    <div className="rounded-lg border border-[#30363d] bg-[#0d1117] p-5">
      <h3 className="text-sm font-semibold text-[#8b949e] uppercase tracking-wider mb-3">
        Regime breakdown (IR per segment)
      </h3>
      <div className="space-y-2">
        {regimes.map((r) => {
          const ir = r.scorecard.ir;
          const widthPct = Math.min(100, (Math.abs(ir) / maxAbs) * 100);
          const isNeg = ir < 0;
          return (
            <div key={r.label} className="text-xs">
              <div className="flex justify-between mb-1">
                <span>{r.label} <span className="text-[#6e7681]">({r.start} → {r.end})</span></span>
                <span className={`font-mono ${ir > 0 ? 'text-green-400' : 'text-red-400'}`}>
                  IR {ir.toFixed(3)} · IC {r.scorecard.ic_mean.toFixed(4)} · n={r.sample_size.toLocaleString()}
                </span>
              </div>
              <div className="h-2 bg-[#21262d] rounded relative overflow-hidden">
                <div
                  className={`h-full ${isNeg ? 'bg-red-500/60' : 'bg-green-500/60'}`}
                  style={{ width: `${widthPct}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Create `AcceptanceLights.tsx`**

```tsx
import type { AcceptanceResult } from './types';

const LABELS: Record<string, string> = {
  ic_mean: 'IC mean ≥ 0.030',
  ir: 'IR ≥ 2.5',
  max_drawdown: '回撤 ≤ 15%',
  daily_turnover: '日换手率 ≤ 20%',
  regimes_all_positive: '所有 regime IR > 0',
};

export function AcceptanceLights({ acceptance }: { acceptance: AcceptanceResult }) {
  return (
    <div className="rounded-lg border border-[#30363d] bg-[#0d1117] p-5">
      <h3 className="text-sm font-semibold text-[#8b949e] uppercase tracking-wider mb-3">
        Acceptance: {acceptance.passed ? (
          <span className="text-green-400">PASS ✅</span>
        ) : (
          <span className="text-red-400">FAIL ❌</span>
        )}
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-2">
        {Object.entries(acceptance.details).map(([key, pass]) => (
          <div key={key} className="flex items-center gap-2 text-xs">
            <span className={pass ? 'text-green-400 text-lg' : 'text-red-400 text-lg'}>●</span>
            <span className="text-[#e6edf3]">{LABELS[key] ?? key}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Replace the detail stub in `Evaluation.tsx`**

Replace the bottom `{selected && ...}` block in `Evaluation.tsx` with:

```tsx
      {selected && <Detail recorderId={selected} />}
```

And add the new component above the default export:

```tsx
import { useEvaluation } from './evaluation/hooks';
import { ScorecardCard } from './evaluation/ScorecardCard';
import { RegimeChart } from './evaluation/RegimeChart';
import { AcceptanceLights } from './evaluation/AcceptanceLights';

function Detail({ recorderId }: { recorderId: string }) {
  const { data, isPending, error } = useEvaluation(recorderId);
  if (isPending) return <div className="text-[#8b949e] text-sm">读取详情…</div>;
  if (error) {
    // 404 = "not evaluated yet"
    return (
      <div className="text-[#8b949e] text-sm">
        该 recorder 还没评估过，点击右侧 "评估" 按钮开始（约 30-90 秒）。
      </div>
    );
  }
  if (!data) return null;
  return (
    <div className="space-y-4">
      <header className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">{data.run_name}</h2>
          <div className="text-xs text-[#8b949e]">
            {data.experiment} · {data.window_start} → {data.window_end} · {data.sample_size.toLocaleString()} 样本
          </div>
        </div>
        <div className="text-xs text-[#6e7681]">
          TopK={data.top_k} · cost={data.cost_bps} bps · 算完于 {data.computed_at.slice(0, 19)}
        </div>
      </header>
      <ScorecardCard scorecard={data.scorecard} />
      <RegimeChart regimes={data.regimes} />
      <AcceptanceLights acceptance={data.acceptance} />
    </div>
  );
}
```

- [ ] **Step 5: Typecheck**

```
cd E:/Projects/qlib/frontend
npx tsc --noEmit
```

- [ ] **Step 6: Smoke test in browser**

Open `/evaluation`. Click a recorder row to select. If you've already clicked "评估" in T8, the Detail section appears with scorecard + regime chart + acceptance lights.

- [ ] **Step 7: Commit**

```
cd E:/Projects/qlib
git add frontend/src/pages/evaluation/ScorecardCard.tsx frontend/src/pages/evaluation/RegimeChart.tsx frontend/src/pages/evaluation/AcceptanceLights.tsx frontend/src/pages/Evaluation.tsx
git commit -m "feat(evaluation): detail view (scorecard + regime + acceptance)"
```

---

### Task 10: Compare mode

**Files:**
- Create: `frontend/src/pages/evaluation/CompareView.tsx`
- Modify: `frontend/src/pages/Evaluation.tsx`

- [ ] **Step 1: Create `CompareView.tsx`**

```tsx
import { useCompare } from './hooks';
import type { RecorderSummary } from './types';
import { ScorecardCard } from './ScorecardCard';

interface Props {
  a: string;
  b: string;
  recorders: RecorderSummary[];
  onClear: () => void;
}

export function CompareView({ a, b, recorders, onClear }: Props) {
  const { data, isPending, error } = useCompare(a, b);

  const nameOf = (id: string) => recorders.find((r) => r.recorder_id === id)?.run_name ?? id.slice(0, 8);

  if (isPending) return <div className="text-[#8b949e] text-sm">比较中… ({nameOf(a)} vs {nameOf(b)})</div>;
  if (error) return <div className="text-red-400 text-sm">比较失败: {String(error)}</div>;
  if (!data) return null;

  return (
    <div className="space-y-4 rounded-lg border border-[#1f6feb] bg-[#0c1f3d] p-5">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">对比 ({nameOf(a)} vs {nameOf(b)})</h2>
        <button onClick={onClear} className="text-xs px-2 py-1 rounded bg-[#21262d] hover:bg-[#30363d] border border-[#30363d]">
          清除对比
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <ScorecardCard scorecard={data.a.scorecard} title={`A: ${data.a.run_name}`} />
        <ScorecardCard scorecard={data.b.scorecard} title={`B: ${data.b.run_name}`} />
      </div>

      <div className="rounded-lg border border-[#30363d] bg-[#0d1117] p-5">
        <h3 className="text-sm font-semibold text-[#8b949e] uppercase tracking-wider mb-3">
          差异 + 配对 t 检验
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
          <Stat label="IC delta (B - A)" value={data.ic_delta.toFixed(4)} positive={data.ic_delta > 0} />
          <Stat label="IR delta (B - A)" value={data.ir_delta.toFixed(3)} positive={data.ir_delta > 0} />
          <Stat label="t-statistic" value={data.paired_t_stat.toFixed(3)} positive={null} />
          <Stat label="p-value" value={data.paired_p_value.toFixed(4)} positive={null} />
        </div>
        <p className="mt-3 text-sm">
          <span className="text-[#8b949e]">结论：</span>
          <span className={
            data.significant_at_05
              ? (data.ic_delta > 0 ? 'text-green-400' : 'text-red-400')
              : 'text-yellow-400'
          }>
            {data.verdict}
          </span>
          {' '}
          <span className="text-[#6e7681]">(α=0.05)</span>
        </p>
      </div>
    </div>
  );
}

function Stat({ label, value, positive }: { label: string; value: string; positive: boolean | null }) {
  const color = positive === null
    ? 'text-[#e6edf3]'
    : positive ? 'text-green-400' : 'text-red-400';
  return (
    <div className="border border-[#30363d] rounded-md px-3 py-2">
      <div className="text-[10px] uppercase tracking-wider text-[#6e7681]">{label}</div>
      <div className={`text-lg font-mono mt-1 ${color}`}>{value}</div>
    </div>
  );
}
```

- [ ] **Step 2: Add compare mode to `Evaluation.tsx`**

Add a "Compare with..." dropdown on each row when one is selected. Simplest: add a "Compare" button next to the existing "评估" button that:
- If no comparison target picked, picks the current row as `b` (and current selected as `a`)
- Triggers `<CompareView a={a} b={b} />` display

Concretely, modify `Evaluation.tsx`:

```tsx
import { CompareView } from './evaluation/CompareView';

// ...inside Evaluation():
const [compareB, setCompareB] = useState<string | null>(null);

// ... in the table, change RecorderRow's onEvaluate handling to also offer compare.
// Simplest UX: a 3rd column-button "比较" that sets compareB to that row's id.
// Since the existing RecorderRow only has one button slot, just add a row-level "Compare with current"
// link when `selected` is set and the row is different from selected.

// Render <CompareView> when both selected and compareB are set:
{selected && compareB && compareB !== selected && (
  <CompareView
    a={selected}
    b={compareB}
    recorders={recorders ?? []}
    onClear={() => setCompareB(null)}
  />
)}
```

Update `RecorderRow.tsx` to add a second small "比较" button next to the existing "评估" button — visible only when `comparable={true}` and onCompare callback wired:

```tsx
// Extend the Props interface:
//   onCompare?: () => void;
//   comparable?: boolean;
// In the cell with the eval button, add:
{props.comparable && (
  <button
    className="ml-2 text-xs px-2 py-1 rounded bg-[#21262d] hover:bg-[#1f6feb] border border-[#30363d]"
    onClick={(e) => {
      e.stopPropagation();
      props.onCompare?.();
    }}
  >
    比较
  </button>
)}
```

And in `Evaluation.tsx` wire `comparable={selected !== null && selected !== rec.recorder_id}` and `onCompare={() => setCompareB(rec.recorder_id)}` per row.

- [ ] **Step 3: Typecheck**

```
cd E:/Projects/qlib/frontend
npx tsc --noEmit
```

- [ ] **Step 4: Smoke test**

Open `/evaluation`. Click row A. Click "比较" on row B. CompareView appears with both scorecards side-by-side + delta + t-test result.

- [ ] **Step 5: Commit**

```
cd E:/Projects/qlib
git add frontend/src/pages/evaluation/CompareView.tsx frontend/src/pages/evaluation/RecorderRow.tsx frontend/src/pages/Evaluation.tsx
git commit -m "feat(evaluation): compare mode with paired t-test view"
```

---

## Phase D — Verification + diagnostic report

### Task 11: Run eval on daily_cn_fresh, write optimization plan

**Files:**
- Create: `docs/superpowers/reports/2026-05-22-current-model-evaluation.md`

- [ ] **Step 1: Run the CLI on daily_cn_fresh**

```
cd E:/Projects/qlib
F:/Tools/Anaconda/envs/qlib/python.exe -m production.eval_recorder f29f042f72634226aa0dc7782d4873d9 --out docs/superpowers/reports
```

(The recorder_id `f29f042f72634226aa0dc7782d4873d9` is the daily_cn_fresh recorder confirmed earlier in the session. If unavailable, use `--list` first to find the actual id.)

Read the generated Markdown report at `docs/superpowers/reports/f29f042f_<ts>.md`.

- [ ] **Step 2: Capture key numbers**

Extract from the report:
- IC mean
- ICIR
- IR (cost-adjusted)
- Max drawdown
- Daily turnover
- Per-regime IRs
- Acceptance: PASS/FAIL

- [ ] **Step 3: Write the optimization plan markdown**

`docs/superpowers/reports/2026-05-22-current-model-evaluation.md`:

```markdown
# Current model evaluation + optimization plan (2026-05-22)

## TL;DR

**`daily_cn_fresh` (current production)**: <IC value> — paste actual numbers here from Step 2.

Acceptance: <PASS / FAIL>. Pipeline implementation complete (CLI + REST + UI). Next: complete β phase training to replace this with the ensemble.

## Methodology

Evaluation pipeline at `backend/app/evaluation/` (described in
`docs/superpowers/plans/2026-05-22-model-evaluation-pipeline.md`). Every metric
follows spec §8 definitions; thresholds per spec §11.

Open-to-open label used for both IC and portfolio metrics (matches what
manual retail traders actually execute).

## Results

### daily_cn_fresh (current production picks driver)

- recorder_id: f29f042f72634226aa0dc7782d4873d9
- prediction window: 2025-01-02 → 2026-05-08
- evaluated samples: <N>

| Metric | Value | Threshold | Pass? |
|---|---|---|---|
| IC mean | <fill> | ≥ 0.030 | <YES/NO> |
| RIC mean | <fill> | — | — |
| ICIR | <fill> | ≥ 0.40 | <YES/NO> |
| Top-bottom monthly | <fill>% | ≥ 1.5% | <YES/NO> |
| Annual return | <fill>% | ≥ +15% | <YES/NO> |
| IR (cost-adj) | <fill> | ≥ 2.5 | <YES/NO> |
| Max drawdown | <fill>% | ≥ -15% | <YES/NO> |
| Daily turnover | <fill>% | ≤ 20% | <YES/NO> |

Per-regime IR: <fill in table>

**Verdict: <PASS/FAIL>**

### rolling_v2_ensemble — N/A

3 partial recorders exist (lgbm_1d, lgbm_5d, lgbm_20d × 1 week each) but:
- No ensemble pred.pkl
- Only 3-day prediction window (insufficient for meaningful IC)
- ALSTM/TRA never trained
- Stacking never fit

Cannot evaluate meaningfully until β phase runs to completion.

## Optimization plan

Priority 1 (immediate): **Run the β phase training to completion**.

- `python -m production.rolling_train run-once --end-date 2026-05-17`
- Expected: 90-150 minutes wall clock (1.5h typical with early stopping)
- Output: rolling_v2_ensemble recorder with ensemble pred.pkl covering the
  test week
- Validate: re-run `python -m production.eval_recorder <new_recorder>` and
  compare with daily_cn_fresh via the Compare UI

Priority 2 (after β has 4+ weeks of data): **Multi-regime backtest**

- The full backtest from 2018 → 2024 across 5 regimes (per spec §8) needs
  the rolling pipeline to backfill historical predictions. Add a CLI
  command `python -m production.rolling_train backfill 2018-01-01..2024-12-31`
  in a follow-up task.

Priority 3 (γ phase trigger): **Add MASTER (Transformer)** once β passes
acceptance for 4 consecutive shadow weeks (per spec §12).

## Validity caveats

1. The current evaluation uses open-to-open labels, but `daily_cn_fresh`
   was trained on close-to-close labels. Its IC measured here will be
   slightly worse than the IC it would score on its training target. The
   portfolio metrics (IR, MDD, turnover) are honest regardless — they
   reflect what a real trader would have made by executing on the open.

2. Regime split for daily_cn_fresh only covers 2025+ (no overlap with the
   2018-2024 spec regimes). Use the "Recent (full window)" segment only;
   the multi-regime acceptance criterion is effectively a single-period
   acceptance for this recorder.

3. Survivorship bias: daily_cn_fresh was trained on *today's* CSI300
   membership. The 2025 prediction window includes stocks that may have
   left the index since. Real-world deployment would have included
   those stocks at the time, so the measured IC slightly overstates
   what production would have achieved.
```

Replace `<fill>` placeholders with actual numbers from Step 2.

- [ ] **Step 4: Manual verification of UI**

- Open `http://127.0.0.1:5173/evaluation` — list view shows all recorders
- Click daily_cn_fresh row, click "评估", wait — should now show:
  - ScorecardCard with 8 metrics + threshold colors
  - RegimeChart with at least the "Recent" segment
  - AcceptanceLights with the 5 criteria status
- Click "比较" on another recorder row (e.g. a β LGBM sub-recorder) — CompareView appears

- [ ] **Step 5: Final commit**

```
cd E:/Projects/qlib
git add docs/superpowers/reports/
git commit -m "docs(eval): baseline measurement of daily_cn_fresh + optimization plan"
```

---

## Acceptance Criteria

Evaluation pipeline is "done" when:

**Functional**

- [ ] `GET /api/evaluation/recorders` returns all recorders across all experiments with summary fields
- [ ] `POST /api/evaluation/run` triggers a fresh evaluation, returns full result; cached on success
- [ ] `GET /api/evaluation/results/{id}` returns cached result; 404 if not computed
- [ ] `GET /api/evaluation/compare?a=&b=` returns side-by-side + paired t-test
- [ ] `python -m production.eval_recorder --list` lists all recorders
- [ ] `python -m production.eval_recorder <id>` writes JSON + Markdown reports
- [ ] `python -m production.eval_recorder --compare A B` prints comparison
- [ ] `/evaluation` page renders list view, detail view, and compare view

**Performance**

- [ ] First eval call on daily_cn_fresh completes within 120 seconds
- [ ] Cached eval call returns within 500ms
- [ ] List view loads within 5 seconds (no full eval performed)

**Quality**

- [ ] All metric thresholds match spec §11 verbatim
- [ ] Color coding (green/red) matches threshold logic for each metric
- [ ] Regime split correctly handles recorders whose prediction window doesn't overlap any spec regime (uses "Recent" catch-all)
- [ ] Acceptance result includes all 5 spec §11 criteria
- [ ] Test suite passes: `pytest backend/app/evaluation/tests/`

**Diagnostic deliverable**

- [ ] `docs/superpowers/reports/2026-05-22-current-model-evaluation.md` contains real measured numbers for daily_cn_fresh + optimization plan

---

## Known Followups (out of this plan's scope)

1. **Persist eval results to sqlite** — currently in-process lru_cache (32 entries). Survives during process lifetime; lost on restart. If users frequently restart the backend, persist to a new `evaluation_results` table.
2. **Background eval queue** — first eval is 30-90s, blocks the request. For Tailscale remote use, consider an async job pattern: POST /run returns a job_id immediately, GET /results polls status.
3. **Historical backfill** — to evaluate against the full 2018-2024 spec regimes, the rolling_train pipeline must first backfill historical predictions.
4. **Per-base-model evaluation** — for rolling_v2_ensemble recorders with 9 base columns (3 models × 3 horizons), add a `view` query param to evaluate each base independently.

---

**End of plan.**
