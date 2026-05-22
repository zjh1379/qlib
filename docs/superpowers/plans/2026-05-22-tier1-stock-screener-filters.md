# Tier 1 Stock Screener Filters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 6 new screener filter dimensions (涨跌幅 N 日 / 振幅 / 量比 / 创 N 日新高 / 板块 / 排除 ST) on top of the existing price + consensus filters, matching the interactive UX of 同花顺 / 东方财富 condition screening — all computed from existing OHLCV + name data without new data sources.

**Architecture:** Backend extracts per-candidate metrics in one batched qlib query (T2), then applies filters in pure-function order (T1 helpers + T4 service logic). Schema gains 6 new optional fields surfaced to the UI. Frontend extracts the filter bar into its own component, syncs all filter state with URL query string for shareable / browser-history-aware filters, debounces slider/text input changes, and shows a live `X / Y` result count + a one-click "重置" button.

**Tech Stack:** Python 3.10 (qlib env `F:/Tools/Anaconda/envs/qlib/python.exe`); FastAPI + SQLAlchemy 2.x async; React 18 + Vite + TanStack Query + Tailwind; openapi-typescript codegen.

**Spec source:** Research dialog 2026-05-22 (THS/EM filter benchmark), §3 (Tier 1 list) + §4 (Interactive logic).

**Run from main repo root** `E:/Projects/qlib` unless a step says otherwise. Backend tests: `cd backend; F:/Tools/Anaconda/envs/qlib/python.exe -m pytest <path>`. Frontend typecheck: `cd frontend; npx tsc --noEmit`. Manual UI verify: backend on 8000, frontend on 5173 (already running).

---

## File Structure

### Backend (`backend/app/`)
- `models/utils.py` (NEW) — pure helpers: `parse_board(symbol) -> "main"|"gem"|"star"|"bj"`, `is_st_name(name) -> bool`, `apply_tier1_filters(items, spec) -> list`
- `core/qlib_adapter.py` — extend with `get_filter_metrics(symbols: list[str], lookback_days: int) -> dict[str, MetricsRow]` returning per-symbol last_close + pct_change_{1,3,5,10,20}d + amplitude + vol_ratio + is_new_high_{20,60,120}d in one batched qlib call
- `models/schemas.py` — extend `ScreenItem` with `pct_change_5d`, `amplitude`, `vol_ratio`, `board`, `is_st` (the most-recent-day-relevant metrics; multi-N metrics fetched server-side and dropped before serialization to keep payload small)
- `models/router.py` — new query params: `pct_change_n` (1/3/5/10/20), `min_pct_change`, `max_pct_change`, `min_amplitude`, `max_amplitude`, `min_vol_ratio`, `max_vol_ratio`, `new_high_n` (0/20/60/120; 0 = off), `boards` (comma-list), `exclude_st` (bool)
- `models/service.py` — fetch metrics for over-fetched candidates, run AND filter pipeline, re-rank
- `models/tests/test_filter_helpers.py` (NEW)
- `models/tests/test_get_filter_metrics.py` (NEW)
- `models/tests/test_screen_filters_integration.py` (NEW)

### Frontend (`frontend/src/`)
- `pages/Picks.tsx` — slimmed down: orchestrate hooks, render `<FilterBar>` + table
- `pages/picks/FilterBar.tsx` (NEW) — all 18+ control inputs organized in labeled groups
- `pages/picks/types.ts` (NEW) — `FilterParams` type, `DEFAULT_FILTERS` constant, `BOARDS` constant
- `pages/picks/useFilterParams.ts` (NEW) — URL-state hook (sync filter state ↔ `useSearchParams`)
- `pages/picks/parse.ts` (NEW) — `parseBoards(csv) -> string[]`, `serializeBoards(boards) -> string`, `parseNumber(str, fallback) -> number`
- `models/hooks.ts` — extend `useScreen` to accept the new params
- `api/client.ts` — extend `screen()` to forward new params
- `api/types.gen.ts` — regenerated

---

## Execution Order

```
Phase A (backend foundation)   T1 → T2 → T3 → T4 → T5
Phase B (frontend hooks/types) T6 → T7 → T8
Phase C (frontend UI)          T9 → T10
Phase D (verification)         T11
```

Each phase end produces a working system: T5 = backend can be exercised via curl; T8 = frontend has the data infrastructure ready; T10 = full UX works; T11 = signed off.

---

## Phase A — Backend

### Task 1: Pure filter helpers (`parse_board`, `is_st_name`)

**Files:**
- Create: `backend/app/models/utils.py`
- Create: `backend/app/models/tests/test_filter_helpers.py`

- [ ] **Step 1: Write failing tests**

`backend/app/models/tests/test_filter_helpers.py`:

```python
import pytest

from app.models.utils import is_st_name, parse_board


@pytest.mark.parametrize(
    "symbol, expected",
    [
        ("SH600519", "main"),   # 沪市主板 (60xxxx)
        ("SZ000001", "main"),   # 深市主板 (00xxxx)
        ("SH601318", "main"),   # 沪市主板 (60xxxx)
        ("SZ300750", "gem"),    # 创业板 (30xxxx)
        ("SH688981", "star"),   # 科创板 (688xxx)
        ("SH689009", "star"),   # 科创板 (689xxx, though 688 is the common prefix)
        ("BJ430047", "bj"),     # 北交所 (BJ exchange)
        ("BJ831010", "bj"),
        ("SH510300", "etf"),    # 沪市 ETF (51xxxx)
        ("SZ159995", "etf"),    # 深市 ETF (15xxxx/16xxxx/17xxxx)
        ("SH588000", "etf"),    # 科创 ETF (588xxx — by convention ETFs even though 58x)
    ],
)
def test_parse_board(symbol, expected):
    assert parse_board(symbol) == expected


def test_parse_board_unknown_returns_other():
    assert parse_board("XX999999") == "other"
    assert parse_board("") == "other"


@pytest.mark.parametrize(
    "name, expected",
    [
        ("ST康美", True),
        ("*ST康美", True),
        ("ST 康美", True),       # trailing/leading whitespace + space variant
        ("ST*康美", True),       # variant ordering
        ("贵州茅台", False),
        ("茅台", False),
        ("", False),
        ("STAR Holdings", False),  # contains "STA" but not ST as a token boundary
    ],
)
def test_is_st_name(name, expected):
    assert is_st_name(name) == expected
```

- [ ] **Step 2: Run, expect failure**

```
cd backend
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/models/tests/test_filter_helpers.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.models.utils'`.

- [ ] **Step 3: Implement the helpers**

`backend/app/models/utils.py`:

```python
"""Pure helpers for Tier 1 screener filters.

These are kept free of qlib / DB dependencies so they can be tested in isolation
and reused from both the service layer (apply filters server-side) and any
future scripts.
"""
from __future__ import annotations

import re

# Code-segment to board mapping per CSRC market structure:
#   60xxxx  / 90xxxx  -> 沪市主板 (Shanghai Main)
#   00xxxx           -> 深市主板 (Shenzhen Main)
#   30xxxx           -> 创业板 (ChiNext / GEM)
#   688xxx / 689xxx  -> 科创板 (STAR Market)
#   430xxx / 8xxxxx  -> 北交所 (Beijing Exchange) -- "BJ" prefix
#   51xxxx / 56xxxx / 58xxxx (excl. 688/689) -> 沪市 ETF
#   15xxxx / 16xxxx / 17xxxx                 -> 深市 ETF
def parse_board(symbol: str) -> str:
    """Classify a qlib-format symbol into 'main' | 'gem' | 'star' | 'bj' | 'etf' | 'other'."""
    if not symbol or len(symbol) < 8:
        return "other"
    prefix = symbol[:2].upper()
    code = symbol[2:]
    if prefix == "BJ":
        return "bj"
    if prefix == "SH":
        if code.startswith(("688", "689")):
            return "star"
        if code.startswith(("60", "90")):
            return "main"
        if code.startswith(("51", "56", "58")):
            return "etf"
    elif prefix == "SZ":
        if code.startswith("30"):
            return "gem"
        if code.startswith(("00",)):
            return "main"
        if code.startswith(("15", "16", "17")):
            return "etf"
    return "other"


# A股 ST patterns: "ST<name>", "*ST<name>", "ST <name>", or with the asterisk
# placed mid-name in some legacy exchanges. We detect the ST token at the start
# of the name (the canonical placement), allowing for optional leading * and
# whitespace.
_ST_PATTERN = re.compile(r"^\s*\*?\s*ST[\s\*]", re.IGNORECASE)


def is_st_name(name: str) -> bool:
    """Return True iff the company name has an ST / *ST risk marker at the start."""
    if not name:
        return False
    # Anchor to start; require ST to be followed by whitespace, asterisk, or
    # end-of-string so we don't falsely match 'STAR' or other ST-prefixed words.
    if _ST_PATTERN.match(name):
        return True
    # Also handle the exact-prefix "ST<chinese>" without a separator — Chinese
    # characters effectively act as a token boundary.
    stripped = name.lstrip()
    if stripped.startswith(("ST", "*ST")):
        rest = stripped[3:] if stripped.startswith("*ST") else stripped[2:]
        if rest and not rest[0].isascii():
            return True
    return False
```

- [ ] **Step 4: Run, expect PASS**

```
cd backend
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/models/tests/test_filter_helpers.py -v
```

Expected: 14 tests PASS (11 board params + 1 unknown + 8 ST params... actually 11+1=12 board, 8 ST = 20 tests).

- [ ] **Step 5: Commit**

```
git add backend/app/models/utils.py backend/app/models/tests/test_filter_helpers.py
git commit -m "feat(screener): parse_board + is_st_name pure helpers"
```

---

### Task 2: `get_filter_metrics` qlib helper

**Files:**
- Modify: `backend/app/core/qlib_adapter.py`
- Create: `backend/app/models/tests/test_get_filter_metrics.py`

- [ ] **Step 1: Write failing tests**

`backend/app/models/tests/test_get_filter_metrics.py`:

```python
"""Tests for app.core.qlib_adapter.get_filter_metrics.

These tests use a real qlib data store (the conda-env cn_data_bs setup) and
sanity-check the shapes + a few computed values. They run only if qlib data is
available; otherwise skip gracefully.
"""
from datetime import date

import pytest

from app.core.qlib_adapter import (
    get_filter_metrics,
    get_latest_close_prices,
    init_qlib_once,
)


@pytest.fixture(scope="module")
def qlib_ready():
    try:
        init_qlib_once()
    except Exception as exc:
        pytest.skip(f"qlib not initializable in this environment: {exc}")


def test_empty_symbols_returns_empty(qlib_ready):
    assert get_filter_metrics([], end_date=None) == {}


def test_known_symbol_shape(qlib_ready):
    # Use a stable, liquid CSI300 symbol that should have data through the
    # qlib data store's last calendar day.
    out = get_filter_metrics(["SH600519"], end_date=None)
    if "SH600519" not in out:
        pytest.skip("SH600519 not present in this qlib data store")
    row = out["SH600519"]
    assert set(row.keys()) >= {
        "last_close",
        "pct_change_1d",
        "pct_change_3d",
        "pct_change_5d",
        "pct_change_10d",
        "pct_change_20d",
        "amplitude",
        "vol_ratio",
        "is_new_high_20d",
        "is_new_high_60d",
        "is_new_high_120d",
    }
    assert row["last_close"] > 0
    # Amplitude is (high - low) / prev_close, typically 0..0.20 for blue chips
    assert 0 <= row["amplitude"] < 1.0


def test_consistency_with_latest_close(qlib_ready):
    syms = ["SH600519", "SH600887"]
    metrics = get_filter_metrics(syms, end_date=None)
    closes = get_latest_close_prices(syms)
    for s in syms:
        if s in metrics and s in closes:
            assert metrics[s]["last_close"] == pytest.approx(closes[s], rel=1e-6)
```

- [ ] **Step 2: Run, expect failure**

```
cd backend
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/models/tests/test_get_filter_metrics.py -v
```

Expected: `ImportError: cannot import name 'get_filter_metrics'`.

- [ ] **Step 3: Implement `get_filter_metrics`**

Append to `backend/app/core/qlib_adapter.py` (after `get_latest_close_prices`):

```python
def get_filter_metrics(
    symbols: list[str],
    end_date: date | None = None,
    lookback_days: int = 200,
) -> dict[str, dict]:
    """Batch-compute Tier 1 filter metrics for a set of symbols in one qlib call.

    Returned dict per symbol:
        {
          "last_close": float,
          "pct_change_1d": float,    # (close_T / close_T-1) - 1
          "pct_change_3d": float,
          "pct_change_5d": float,
          "pct_change_10d": float,
          "pct_change_20d": float,
          "amplitude": float,        # (high_T - low_T) / close_T-1
          "vol_ratio": float,        # vol_T / mean(vol_T-1..T-5)
          "is_new_high_20d": bool,   # close_T == max(close, T-19..T)
          "is_new_high_60d": bool,
          "is_new_high_120d": bool,
        }

    Symbols with insufficient history (e.g. listed within `lookback_days`) are
    returned with NaN-safe defaults: the relevant pct_change_N becomes 0 and
    new-high flags become False. last_close is always populated when the symbol
    has any OHLCV in the window; otherwise the symbol is omitted from the result.
    """
    init_qlib_once()
    if not symbols:
        return {}
    if end_date is None:
        end_date = get_calendar_end()
    start_date = end_date - timedelta(days=lookback_days)

    try:
        df = D.features(
            instruments=symbols,
            fields=["$open", "$high", "$low", "$close", "$volume"],
            start_time=start_date.isoformat(),
            end_time=end_date.isoformat(),
        )
    except Exception as exc:
        _log.warning("filter_metrics_fetch_failed error=%s", str(exc))
        return {}
    if df is None or df.empty:
        return {}

    out: dict[str, dict] = {}
    for inst, group in df.groupby(level="instrument"):
        g = group.dropna(subset=["$close"])
        if g.empty:
            continue
        closes = g["$close"].to_numpy()
        highs = g["$high"].to_numpy()
        lows = g["$low"].to_numpy()
        vols = g["$volume"].to_numpy()
        last = closes[-1]

        def _pct_n(n: int) -> float:
            if len(closes) <= n:
                return 0.0
            prev = closes[-1 - n]
            return float((last / prev) - 1.0) if prev > 0 else 0.0

        amp = 0.0
        if len(closes) >= 2 and closes[-2] > 0:
            amp = float((highs[-1] - lows[-1]) / closes[-2])

        vol_ratio = 0.0
        if len(vols) >= 6:
            past5_mean = vols[-6:-1].mean()
            if past5_mean > 0:
                vol_ratio = float(vols[-1] / past5_mean)

        def _is_new_high(n: int) -> bool:
            if len(closes) < n:
                return False
            window = closes[-n:]
            # Use a tiny epsilon to absorb float noise
            return bool(last + 1e-9 >= window.max())

        out[inst] = {
            "last_close": float(last),
            "pct_change_1d": _pct_n(1),
            "pct_change_3d": _pct_n(3),
            "pct_change_5d": _pct_n(5),
            "pct_change_10d": _pct_n(10),
            "pct_change_20d": _pct_n(20),
            "amplitude": amp,
            "vol_ratio": vol_ratio,
            "is_new_high_20d": _is_new_high(20),
            "is_new_high_60d": _is_new_high(60),
            "is_new_high_120d": _is_new_high(120),
        }
    return out
```

- [ ] **Step 4: Run, expect PASS**

```
cd backend
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/models/tests/test_get_filter_metrics.py -v
```

Expected: 3 tests PASS (or SKIPPED on machines without qlib data — that's acceptable).

- [ ] **Step 5: Commit**

```
git add backend/app/core/qlib_adapter.py backend/app/models/tests/test_get_filter_metrics.py
git commit -m "feat(screener): get_filter_metrics batch qlib helper"
```

---

### Task 3: Extend `ScreenItem` schema with derived fields

**Files:**
- Modify: `backend/app/models/schemas.py`

- [ ] **Step 1: Add fields**

In `backend/app/models/schemas.py`, replace the `ScreenItem` class with:

```python
class ScreenItem(BaseModel):
    rank: int
    symbol: str
    name: str = ""
    score_today: float
    score_avg: float
    rank_avg: float
    days_in_top: int
    consensus: float = 0.0
    base_scores: dict[str, float] = Field(default_factory=dict)
    last_price: float | None = None

    # Tier 1 screener metrics (T3) — exposed to the UI for at-a-glance display
    # and to drive client-side highlights. Server-side multi-N pct_change values
    # used by filters are NOT serialized; only the canonical 5d is surfaced.
    pct_change_5d: float | None = None
    amplitude: float | None = None
    vol_ratio: float | None = None
    board: str | None = None  # main | gem | star | bj | etf | other
    is_st: bool = False
```

- [ ] **Step 2: Verify the schema parses by running existing tests**

```
cd backend
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/models/tests/ -v -k "not test_get_filter_metrics"
```

Expected: all existing tests still PASS (the new fields are optional with safe defaults).

- [ ] **Step 3: Commit**

```
git add backend/app/models/schemas.py
git commit -m "feat(screener): ScreenItem gains pct_change_5d/amplitude/vol_ratio/board/is_st"
```

---

### Task 4: Filter query params + service filter logic

**Files:**
- Modify: `backend/app/models/router.py`
- Modify: `backend/app/models/service.py`
- Modify: `backend/app/models/utils.py` (add `apply_tier1_filters`)

- [ ] **Step 1: Extend the router**

Replace the `screen` route in `backend/app/models/router.py`:

```python
@router.get("/screen", response_model=ScreenResponse)
def screen(
    top: int = Query(default=30, ge=1, le=300),
    days: int = Query(default=5, ge=1, le=60),
    min_top: int = Query(default=0, ge=0),
    experiment: str | None = Query(default=None),
    view: str = Query(default="ensemble", pattern="^(ensemble|lightgbm|alstm|tra)$"),
    # Existing price filter
    min_price: float | None = Query(default=None, ge=0),
    max_price: float | None = Query(default=None, ge=0),
    # Tier 1 filters (T4)
    pct_change_n: int = Query(default=5, description="Lookback in trading days for pct_change filter", pattern=None),
    min_pct_change: float | None = Query(default=None, description="Min pct change over pct_change_n (e.g. 0.05 = +5%)"),
    max_pct_change: float | None = Query(default=None),
    min_amplitude: float | None = Query(default=None, ge=0),
    max_amplitude: float | None = Query(default=None, ge=0),
    min_vol_ratio: float | None = Query(default=None, ge=0),
    max_vol_ratio: float | None = Query(default=None, ge=0),
    new_high_n: int = Query(default=0, description="0=off, 20/60/120 = require close to be N-day high"),
    boards: str | None = Query(default=None, description="Comma list: main,gem,star,bj,etf"),
    exclude_st: bool = Query(default=True),
):
    if pct_change_n not in (1, 3, 5, 10, 20):
        raise BusinessError(
            code="bad_pct_change_n",
            message="pct_change_n must be one of 1,3,5,10,20",
            http_status=400,
        )
    if new_high_n not in (0, 20, 60, 120):
        raise BusinessError(
            code="bad_new_high_n",
            message="new_high_n must be one of 0,20,60,120",
            http_status=400,
        )
    return service.screen(
        top=top, days=days, min_top=min_top, experiment=experiment, view=view,
        min_price=min_price, max_price=max_price,
        pct_change_n=pct_change_n,
        min_pct_change=min_pct_change, max_pct_change=max_pct_change,
        min_amplitude=min_amplitude, max_amplitude=max_amplitude,
        min_vol_ratio=min_vol_ratio, max_vol_ratio=max_vol_ratio,
        new_high_n=new_high_n,
        boards=boards,
        exclude_st=exclude_st,
    )
```

Add the import at the top:

```python
from app.core.exceptions import BusinessError
```

- [ ] **Step 2: Add `apply_tier1_filters` to utils.py**

Append to `backend/app/models/utils.py`:

```python
from dataclasses import dataclass


@dataclass
class Tier1FilterSpec:
    pct_change_n: int = 5
    min_pct_change: float | None = None
    max_pct_change: float | None = None
    min_amplitude: float | None = None
    max_amplitude: float | None = None
    min_vol_ratio: float | None = None
    max_vol_ratio: float | None = None
    new_high_n: int = 0  # 0 = off
    boards: set[str] | None = None  # None = no board filter; set means OR within boards
    exclude_st: bool = True


def _passes_range(value: float | None, lo: float | None, hi: float | None) -> bool:
    """Inclusive range check that treats either bound as 'unbounded' when None.
    A None value (e.g. metric not available) fails any non-None bound."""
    if lo is None and hi is None:
        return True
    if value is None:
        return False
    if lo is not None and value < lo:
        return False
    if hi is not None and value > hi:
        return False
    return True


def apply_tier1_filters(
    rows: list[dict],
    spec: Tier1FilterSpec,
) -> list[dict]:
    """Apply Tier 1 filters with AND semantics. Each row must carry the metric
    fields produced by qlib_adapter.get_filter_metrics + a 'symbol' + 'name' + 'board'.

    Returns the rows that pass every filter, in input order.
    """
    out: list[dict] = []
    pct_key = f"pct_change_{spec.pct_change_n}d"
    for r in rows:
        if not _passes_range(r.get(pct_key), spec.min_pct_change, spec.max_pct_change):
            continue
        if not _passes_range(r.get("amplitude"), spec.min_amplitude, spec.max_amplitude):
            continue
        if not _passes_range(r.get("vol_ratio"), spec.min_vol_ratio, spec.max_vol_ratio):
            continue
        if spec.new_high_n != 0:
            key = f"is_new_high_{spec.new_high_n}d"
            if not r.get(key, False):
                continue
        if spec.boards is not None and len(spec.boards) > 0:
            if r.get("board") not in spec.boards:
                continue
        if spec.exclude_st and r.get("is_st", False):
            continue
        out.append(r)
    return out
```

- [ ] **Step 3: Wire metrics + filtering into `service.screen`**

In `backend/app/models/service.py`, replace the existing `screen` function:

```python
def screen(
    top: int = 30,
    days: int = 5,
    min_top: int = 0,
    experiment: str | None = None,
    view: str = "ensemble",
    min_price: float | None = None,
    max_price: float | None = None,
    pct_change_n: int = 5,
    min_pct_change: float | None = None,
    max_pct_change: float | None = None,
    min_amplitude: float | None = None,
    max_amplitude: float | None = None,
    min_vol_ratio: float | None = None,
    max_vol_ratio: float | None = None,
    new_high_n: int = 0,
    boards: str | None = None,
    exclude_st: bool = True,
) -> dict:
    """Rank + filter the model's universe.

    Filter pipeline (AND semantics, applied to over-fetched candidates):
      1. Existing: price range
      2. Tier 1: pct_change_{pct_change_n}d ∈ [min,max]
      3. Tier 1: amplitude ∈ [min,max]
      4. Tier 1: vol_ratio ∈ [min,max]
      5. Tier 1: is_new_high_{new_high_n}d == True (if new_high_n != 0)
      6. Tier 1: board ∈ boards (OR within multiselect)
      7. Tier 1: not is_st (if exclude_st)

    Symbols missing a metric fail any non-trivial bound on that metric.
    Returned items are re-ranked 1..N after filtering.
    """
    from app.core.qlib_adapter import get_filter_metrics
    from app.models.utils import Tier1FilterSpec, apply_tier1_filters, is_st_name, parse_board

    init_qlib_once()
    s = Settings()
    exp = experiment or s.default_experiment
    recorder_id = get_latest_recorder_id(exp)
    pred = load_pred(recorder_id, experiment_name=exp)

    if isinstance(pred, pd.Series):
        df = pred.to_frame(name="score")
    else:
        df = pred.copy()
        if "score" not in df.columns:
            if df.shape[1] == 1:
                df = df.rename(columns={df.columns[0]: "score"})
            else:
                raise ValueError("pred frame missing 'score' column")

    if df.index.names != ["datetime", "instrument"]:
        df.index = df.index.set_names(["datetime", "instrument"])

    if view != "ensemble":
        _view_prefix = {"lightgbm": "lgbm_", "alstm": "alstm_", "tra": "tra_"}
        prefix = _view_prefix.get(view, f"{view}_")
        view_cols = [c for c in df.columns if c.startswith(prefix)]
        if view_cols:
            df = df.copy()
            df["score"] = df[view_cols].mean(axis=1)

    dates = df.index.get_level_values("datetime").unique().sort_values()
    today = dates[-1]
    last_slice = df.xs(today, level="datetime")
    universe_size = int(last_slice["score"].count())

    name_map = _name_map()

    # Decide whether any expensive filter is active so we know whether to
    # over-fetch and run the metric pipeline.
    tier1_active = (
        min_price is not None or max_price is not None
        or min_pct_change is not None or max_pct_change is not None
        or min_amplitude is not None or max_amplitude is not None
        or min_vol_ratio is not None or max_vol_ratio is not None
        or new_high_n != 0
        or boards is not None
        or exclude_st
    )
    fetch_top = min(top * 4, 300) if tier1_active else top
    items = _build_screen_items(df, top=fetch_top, days=days, min_top=min_top, name_map=name_map)

    if items:
        prices = get_latest_close_prices([it.symbol for it in items])
        for it in items:
            it.last_price = prices.get(it.symbol)

        # Compute metrics + board + ST flags for every candidate
        metrics = get_filter_metrics([it.symbol for it in items])
        for it in items:
            m = metrics.get(it.symbol, {})
            it.pct_change_5d = m.get("pct_change_5d")
            it.amplitude = m.get("amplitude")
            it.vol_ratio = m.get("vol_ratio")
            it.board = parse_board(it.symbol)
            it.is_st = is_st_name(it.name)
            # Stash multi-N metrics in a transient dict so the filter pipeline
            # can read them without needing per-N schema fields.
            it.__dict__["_metrics"] = m

    # Build filter spec
    boards_set: set[str] | None = None
    if boards:
        boards_set = {b.strip() for b in boards.split(",") if b.strip()}

    spec = Tier1FilterSpec(
        pct_change_n=pct_change_n,
        min_pct_change=min_pct_change, max_pct_change=max_pct_change,
        min_amplitude=min_amplitude, max_amplitude=max_amplitude,
        min_vol_ratio=min_vol_ratio, max_vol_ratio=max_vol_ratio,
        new_high_n=new_high_n,
        boards=boards_set,
        exclude_st=exclude_st,
    )

    # Translate items -> dicts that apply_tier1_filters expects
    rows: list[dict] = []
    for it in items:
        m = it.__dict__.get("_metrics", {})
        rows.append({
            "symbol": it.symbol,
            "name": it.name,
            "board": it.board,
            "is_st": it.is_st,
            "amplitude": it.amplitude,
            "vol_ratio": it.vol_ratio,
            "pct_change_1d": m.get("pct_change_1d"),
            "pct_change_3d": m.get("pct_change_3d"),
            "pct_change_5d": it.pct_change_5d,
            "pct_change_10d": m.get("pct_change_10d"),
            "pct_change_20d": m.get("pct_change_20d"),
            "is_new_high_20d": m.get("is_new_high_20d", False),
            "is_new_high_60d": m.get("is_new_high_60d", False),
            "is_new_high_120d": m.get("is_new_high_120d", False),
            "_item": it,
        })

    # Existing price filter (kept in service, not Tier1FilterSpec)
    if min_price is not None or max_price is not None:
        rows = [r for r in rows if _passes_price(r["_item"].last_price, min_price, max_price)]

    # Tier 1 filters
    rows = apply_tier1_filters(rows, spec)

    # Re-rank and trim
    filtered_items = [r["_item"] for r in rows[:top]]
    for new_rank, it in enumerate(filtered_items, start=1):
        it.rank = new_rank

    # Strip transient metrics dict before serialization
    for it in filtered_items:
        it.__dict__.pop("_metrics", None)

    return {
        "experiment": exp,
        "recorder_id": recorder_id,
        "latest_date": str(today.date()),
        "window_days": days,
        "universe_size": universe_size,
        "items": [it.model_dump() for it in filtered_items],
    }


def _passes_price(price: float | None, lo: float | None, hi: float | None) -> bool:
    if lo is None and hi is None:
        return True
    if price is None:
        return False
    if lo is not None and price < lo:
        return False
    if hi is not None and price > hi:
        return False
    return True
```

- [ ] **Step 4: Smoke-test via direct service call**

```
cd backend
F:/Tools/Anaconda/envs/qlib/python.exe -c "
from app.models.service import screen
r = screen(top=5, exclude_st=False)
for it in r['items']:
    print(f'  rank={it[\"rank\"]} {it[\"symbol\"]} board={it[\"board\"]} is_st={it[\"is_st\"]} amp={it[\"amplitude\"]} vol_ratio={it[\"vol_ratio\"]} pct5d={it[\"pct_change_5d\"]}')
"
```

Expected: 5 items, all with populated board/is_st/amplitude/vol_ratio/pct_change_5d fields.

- [ ] **Step 5: Commit**

```
git add backend/app/models/router.py backend/app/models/service.py backend/app/models/utils.py
git commit -m "feat(screener): Tier 1 filter params + service pipeline + AND logic"
```

---

### Task 5: Integration test — full filter pipeline

**Files:**
- Create: `backend/app/models/tests/test_screen_filters_integration.py`

- [ ] **Step 1: Write the integration test**

`backend/app/models/tests/test_screen_filters_integration.py`:

```python
"""End-to-end tests for the screener filter pipeline via the FastAPI test client.

These hit the real qlib data store and the real default mlruns recorder; they
skip if either is unavailable.
"""
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app


@pytest.fixture
async def client():
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        async with app.router.lifespan_context(app):
            yield c


async def _try_screen(client, **params) -> dict:
    qs = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
    r = await client.get(f"/api/models/screen?{qs}")
    return r.status_code, r.json()


async def test_baseline_returns_items_with_new_fields(client):
    status, body = await _try_screen(client, top=3, exclude_st="false")
    if status == 404:
        pytest.skip("no recorder in default experiment")
    assert status == 200
    assert len(body["items"]) <= 3
    for it in body["items"]:
        # New Tier 1 fields are present (may be None when qlib data missing)
        for k in ("board", "is_st", "amplitude", "vol_ratio", "pct_change_5d"):
            assert k in it


async def test_board_filter_main_only(client):
    status, body = await _try_screen(client, top=20, boards="main", exclude_st="false")
    if status == 404:
        pytest.skip("no recorder")
    assert status == 200
    for it in body["items"]:
        assert it["board"] == "main"


async def test_exclude_st_default_drops_st(client):
    status, body = await _try_screen(client, top=30, exclude_st="true")
    if status == 404:
        pytest.skip("no recorder")
    assert status == 200
    for it in body["items"]:
        assert it["is_st"] is False


async def test_bad_pct_change_n_returns_400(client):
    status, body = await _try_screen(client, top=5, pct_change_n=7)
    assert status == 400
    assert body["code"] == "bad_pct_change_n"


async def test_bad_new_high_n_returns_400(client):
    status, body = await _try_screen(client, top=5, new_high_n=30)
    assert status == 400
    assert body["code"] == "bad_new_high_n"


async def test_filtered_items_renumbered_from_1(client):
    status, body = await _try_screen(client, top=10, boards="main", exclude_st="true")
    if status == 404:
        pytest.skip("no recorder")
    assert status == 200
    items = body["items"]
    if items:
        # Ranks must be 1..N contiguous after filtering
        ranks = [it["rank"] for it in items]
        assert ranks == list(range(1, len(items) + 1))
```

- [ ] **Step 2: Run, expect PASS (or SKIPPED where data absent)**

```
cd backend
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest app/models/tests/test_screen_filters_integration.py -v
```

Expected: 6 tests PASS (or some SKIPPED on data-less CI).

- [ ] **Step 3: Smoke-test via curl with a running backend**

Assumes the dev backend is already up on :8000 (started earlier in the session):

```
F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -c "
import urllib.request, json
r = json.loads(urllib.request.urlopen('http://127.0.0.1:8000/api/models/screen?top=10&boards=main&exclude_st=true&min_pct_change=0', timeout=30).read())
print(f'returned {len(r[\"items\"])} items, all main board, all up >0% (5d), all non-ST:')
for it in r['items']:
    print(f'  {it[\"symbol\"]} {it[\"name\"]} board={it[\"board\"]} st={it[\"is_st\"]} pct5d={it[\"pct_change_5d\"]:.4f}')
"
```

- [ ] **Step 4: Commit**

```
git add backend/app/models/tests/test_screen_filters_integration.py
git commit -m "test(screener): integration tests for Tier 1 filter pipeline"
```

---

## Phase B — Frontend hooks/types

### Task 6: Regen types + extend `client` + `hooks`

**Files:**
- Modify: `frontend/src/api/types.gen.ts` (regenerated)
- Modify: `frontend/src/api/client.ts`
- Modify: `frontend/src/models/hooks.ts`
- Create: `frontend/src/pages/picks/types.ts`

- [ ] **Step 1: Regenerate OpenAPI types**

```
cd frontend
npm run gen:api
```

Verify the new params landed:

```
grep -n "pct_change_n\|min_amplitude\|new_high_n\|boards\|exclude_st" frontend/src/api/types.gen.ts | head -10
```

Expected: each appears at least once.

- [ ] **Step 2: Create the shared types module**

`frontend/src/pages/picks/types.ts`:

```typescript
export type View = 'ensemble' | 'lightgbm' | 'alstm' | 'tra';
export type Board = 'main' | 'gem' | 'star' | 'bj' | 'etf';
export type PctChangeN = 1 | 3 | 5 | 10 | 20;
export type NewHighN = 0 | 20 | 60 | 120;

export const PCT_CHANGE_N_OPTIONS: PctChangeN[] = [1, 3, 5, 10, 20];
export const NEW_HIGH_N_OPTIONS: NewHighN[] = [0, 20, 60, 120];

export const BOARDS: { value: Board; label: string }[] = [
  { value: 'main', label: '主板' },
  { value: 'gem', label: '创业板' },
  { value: 'star', label: '科创板' },
  { value: 'bj', label: '北交所' },
  { value: 'etf', label: 'ETF' },
];

export interface FilterParams {
  // Core (existing)
  top: number;
  days: number;
  min_top: number;
  view: View;
  min_price: number | null;
  max_price: number | null;
  // Tier 1
  pct_change_n: PctChangeN;
  min_pct_change: number | null;
  max_pct_change: number | null;
  min_amplitude: number | null;
  max_amplitude: number | null;
  min_vol_ratio: number | null;
  max_vol_ratio: number | null;
  new_high_n: NewHighN;
  boards: Board[];   // empty = no filter
  exclude_st: boolean;
  // UI-only (not sent to backend)
  min_consensus: number;
}

export const DEFAULT_FILTERS: FilterParams = {
  top: 30,
  days: 5,
  min_top: 0,
  view: 'ensemble',
  min_price: null,
  max_price: 30,
  pct_change_n: 5,
  min_pct_change: null,
  max_pct_change: null,
  min_amplitude: null,
  max_amplitude: null,
  min_vol_ratio: null,
  max_vol_ratio: null,
  new_high_n: 0,
  boards: [],
  exclude_st: true,
  min_consensus: 0,
};
```

- [ ] **Step 3: Extend the API client**

In `frontend/src/api/client.ts`, replace the `screen:` function body inside `models:`:

```typescript
    screen: (
      params: {
        top?: number;
        days?: number;
        min_top?: number;
        experiment?: string;
        view?: 'ensemble' | 'lightgbm' | 'alstm' | 'tra';
        min_price?: number | null;
        max_price?: number | null;
        pct_change_n?: 1 | 3 | 5 | 10 | 20;
        min_pct_change?: number | null;
        max_pct_change?: number | null;
        min_amplitude?: number | null;
        max_amplitude?: number | null;
        min_vol_ratio?: number | null;
        max_vol_ratio?: number | null;
        new_high_n?: 0 | 20 | 60 | 120;
        boards?: string[]; // serialized as comma list
        exclude_st?: boolean;
      } = {},
    ) => {
      type R = paths['/api/models/screen']['get']['responses']['200']['content']['application/json'];
      const q = new URLSearchParams();
      const setNullable = (key: string, v: number | null | undefined) => {
        if (v !== undefined && v !== null) q.set(key, String(v));
      };
      if (params.top !== undefined) q.set('top', String(params.top));
      if (params.days !== undefined) q.set('days', String(params.days));
      if (params.min_top !== undefined) q.set('min_top', String(params.min_top));
      if (params.experiment) q.set('experiment', params.experiment);
      if (params.view) q.set('view', params.view);
      setNullable('min_price', params.min_price);
      setNullable('max_price', params.max_price);
      if (params.pct_change_n !== undefined) q.set('pct_change_n', String(params.pct_change_n));
      setNullable('min_pct_change', params.min_pct_change);
      setNullable('max_pct_change', params.max_pct_change);
      setNullable('min_amplitude', params.min_amplitude);
      setNullable('max_amplitude', params.max_amplitude);
      setNullable('min_vol_ratio', params.min_vol_ratio);
      setNullable('max_vol_ratio', params.max_vol_ratio);
      if (params.new_high_n !== undefined) q.set('new_high_n', String(params.new_high_n));
      if (params.boards && params.boards.length > 0) q.set('boards', params.boards.join(','));
      if (params.exclude_st !== undefined) q.set('exclude_st', params.exclude_st ? 'true' : 'false');
      const qs = q.toString();
      return request<R>(`/api/models/screen${qs ? '?' + qs : ''}`);
    },
```

- [ ] **Step 4: Extend the hook**

Replace `useScreen` in `frontend/src/models/hooks.ts`:

```typescript
import { useQuery } from '@tanstack/react-query';
import { api } from '@/api/client';

export function useScreen(
  params: {
    top?: number;
    days?: number;
    min_top?: number;
    view?: 'ensemble' | 'lightgbm' | 'alstm' | 'tra';
    min_price?: number | null;
    max_price?: number | null;
    pct_change_n?: 1 | 3 | 5 | 10 | 20;
    min_pct_change?: number | null;
    max_pct_change?: number | null;
    min_amplitude?: number | null;
    max_amplitude?: number | null;
    min_vol_ratio?: number | null;
    max_vol_ratio?: number | null;
    new_high_n?: 0 | 20 | 60 | 120;
    boards?: string[];
    exclude_st?: boolean;
  } = {},
) {
  return useQuery({
    queryKey: ['models', 'screen', params],
    queryFn: () => api.models.screen(params),
    staleTime: 5 * 60_000,
    placeholderData: (prev) => prev, // keep old results visible during refetch
  });
}

export function usePredictionHistory(symbol: string, days = 60) {
  return useQuery({
    queryKey: ['models', 'predictions', symbol, days],
    queryFn: () => api.models.predictions(symbol, { days }),
    enabled: !!symbol,
    staleTime: 5 * 60_000,
  });
}
```

- [ ] **Step 5: Typecheck**

```
cd frontend
npx tsc --noEmit
```

Expected: 0 errors.

- [ ] **Step 6: Commit**

```
git add frontend/src/api/types.gen.ts frontend/src/api/client.ts frontend/src/models/hooks.ts frontend/src/pages/picks/types.ts
git commit -m "feat(screener): regen types + client/hooks for Tier 1 filters"
```

---

### Task 7: URL-state hook `useFilterParams`

**Files:**
- Create: `frontend/src/pages/picks/parse.ts`
- Create: `frontend/src/pages/picks/useFilterParams.ts`

- [ ] **Step 1: Create the parse helpers**

`frontend/src/pages/picks/parse.ts`:

```typescript
import type { Board, FilterParams, NewHighN, PctChangeN, View } from './types';
import { BOARDS, DEFAULT_FILTERS, NEW_HIGH_N_OPTIONS, PCT_CHANGE_N_OPTIONS } from './types';

const BOARD_VALUES: readonly Board[] = BOARDS.map((b) => b.value);
const VIEWS: readonly View[] = ['ensemble', 'lightgbm', 'alstm', 'tra'];

export function parseInt32(raw: string | null, fallback: number): number {
  if (raw === null || raw === '') return fallback;
  const n = Number.parseInt(raw, 10);
  return Number.isFinite(n) ? n : fallback;
}

export function parseFloat32(raw: string | null, fallback: number | null): number | null {
  if (raw === null || raw === '') return fallback;
  const n = Number.parseFloat(raw);
  return Number.isFinite(n) ? n : fallback;
}

export function parseBool(raw: string | null, fallback: boolean): boolean {
  if (raw === null) return fallback;
  return raw === 'true' || raw === '1';
}

export function parseEnum<T extends string | number>(
  raw: string | null,
  allowed: readonly T[],
  fallback: T,
): T {
  if (raw === null || raw === '') return fallback;
  const parsed: string | number = typeof allowed[0] === 'number' ? Number(raw) : raw;
  return (allowed as readonly (string | number)[]).includes(parsed) ? (parsed as T) : fallback;
}

export function parseBoards(raw: string | null): Board[] {
  if (raw === null || raw === '') return [];
  return raw
    .split(',')
    .map((s) => s.trim())
    .filter((s): s is Board => (BOARD_VALUES as readonly string[]).includes(s));
}

export function serializeBoards(boards: Board[]): string {
  return boards.join(',');
}

/** Parse a URLSearchParams instance into FilterParams, falling back to DEFAULT_FILTERS for any missing key. */
export function paramsFromUrl(sp: URLSearchParams): FilterParams {
  return {
    top: parseInt32(sp.get('top'), DEFAULT_FILTERS.top),
    days: parseInt32(sp.get('days'), DEFAULT_FILTERS.days),
    min_top: parseInt32(sp.get('min_top'), DEFAULT_FILTERS.min_top),
    view: parseEnum(sp.get('view'), VIEWS, DEFAULT_FILTERS.view),
    min_price: parseFloat32(sp.get('min_price'), DEFAULT_FILTERS.min_price),
    max_price: parseFloat32(sp.get('max_price'), DEFAULT_FILTERS.max_price),
    pct_change_n: parseEnum<PctChangeN>(
      sp.get('pct_change_n'),
      PCT_CHANGE_N_OPTIONS,
      DEFAULT_FILTERS.pct_change_n,
    ),
    min_pct_change: parseFloat32(sp.get('min_pct_change'), DEFAULT_FILTERS.min_pct_change),
    max_pct_change: parseFloat32(sp.get('max_pct_change'), DEFAULT_FILTERS.max_pct_change),
    min_amplitude: parseFloat32(sp.get('min_amplitude'), DEFAULT_FILTERS.min_amplitude),
    max_amplitude: parseFloat32(sp.get('max_amplitude'), DEFAULT_FILTERS.max_amplitude),
    min_vol_ratio: parseFloat32(sp.get('min_vol_ratio'), DEFAULT_FILTERS.min_vol_ratio),
    max_vol_ratio: parseFloat32(sp.get('max_vol_ratio'), DEFAULT_FILTERS.max_vol_ratio),
    new_high_n: parseEnum<NewHighN>(
      sp.get('new_high_n'),
      NEW_HIGH_N_OPTIONS,
      DEFAULT_FILTERS.new_high_n,
    ),
    boards: parseBoards(sp.get('boards')),
    exclude_st: parseBool(sp.get('exclude_st'), DEFAULT_FILTERS.exclude_st),
    min_consensus: parseFloat32(sp.get('min_consensus'), DEFAULT_FILTERS.min_consensus) ?? 0,
  };
}

/** Inverse: write only non-default keys back into the URL. */
export function urlFromParams(p: FilterParams): URLSearchParams {
  const sp = new URLSearchParams();
  const setIfChanged = (key: keyof FilterParams, val: unknown) => {
    if (val === null || val === undefined) return;
    if (val === DEFAULT_FILTERS[key]) return;
    if (Array.isArray(val)) {
      if (val.length === 0) return;
      sp.set(key, val.join(','));
    } else if (typeof val === 'boolean') {
      sp.set(key, val ? 'true' : 'false');
    } else {
      sp.set(key, String(val));
    }
  };
  setIfChanged('top', p.top);
  setIfChanged('days', p.days);
  setIfChanged('min_top', p.min_top);
  setIfChanged('view', p.view);
  setIfChanged('min_price', p.min_price);
  setIfChanged('max_price', p.max_price);
  setIfChanged('pct_change_n', p.pct_change_n);
  setIfChanged('min_pct_change', p.min_pct_change);
  setIfChanged('max_pct_change', p.max_pct_change);
  setIfChanged('min_amplitude', p.min_amplitude);
  setIfChanged('max_amplitude', p.max_amplitude);
  setIfChanged('min_vol_ratio', p.min_vol_ratio);
  setIfChanged('max_vol_ratio', p.max_vol_ratio);
  setIfChanged('new_high_n', p.new_high_n);
  setIfChanged('boards', p.boards);
  setIfChanged('exclude_st', p.exclude_st);
  setIfChanged('min_consensus', p.min_consensus);
  return sp;
}
```

- [ ] **Step 2: Create the hook**

`frontend/src/pages/picks/useFilterParams.ts`:

```typescript
import { useCallback, useMemo } from 'react';
import { useSearchParams } from 'react-router-dom';

import { DEFAULT_FILTERS, FilterParams } from './types';
import { paramsFromUrl, urlFromParams } from './parse';

/** Two-way bind FilterParams ↔ URL query string. Reading is O(1), writing
 *  replaces the URL via react-router's `setSearchParams` (no full reload). */
export function useFilterParams(): [FilterParams, (next: Partial<FilterParams>) => void, () => void] {
  const [sp, setSp] = useSearchParams();
  const params = useMemo(() => paramsFromUrl(sp), [sp]);

  const update = useCallback(
    (patch: Partial<FilterParams>) => {
      const merged: FilterParams = { ...params, ...patch };
      const nextSp = urlFromParams(merged);
      setSp(nextSp, { replace: false });
    },
    [params, setSp],
  );

  const reset = useCallback(() => {
    setSp(urlFromParams(DEFAULT_FILTERS), { replace: false });
  }, [setSp]);

  return [params, update, reset];
}
```

- [ ] **Step 3: Typecheck**

```
cd frontend
npx tsc --noEmit
```

Expected: 0 errors.

- [ ] **Step 4: Commit**

```
git add frontend/src/pages/picks/parse.ts frontend/src/pages/picks/useFilterParams.ts
git commit -m "feat(screener): URL-state hook + parse helpers for filter params"
```

---

### Task 8: Extract `FilterBar` component (no UI change yet)

**Files:**
- Create: `frontend/src/pages/picks/FilterBar.tsx`
- Modify: `frontend/src/pages/Picks.tsx`

- [ ] **Step 1: Create FilterBar with the current set of controls (no new filters yet)**

`frontend/src/pages/picks/FilterBar.tsx`:

```typescript
import type { FilterParams } from './types';

interface FilterBarProps {
  params: FilterParams;
  resultCount: number | null;       // null = loading
  candidateCount: number | null;    // null = loading
  onChange: (patch: Partial<FilterParams>) => void;
  onReset: () => void;
}

export function FilterBar({ params, resultCount, candidateCount, onChange, onReset }: FilterBarProps) {
  return (
    <div className="rounded-lg border border-[#30363d] bg-[#0d1117] p-5 space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-[#8b949e] uppercase tracking-wider">筛选</h2>
        <div className="flex items-center gap-3">
          <span className="text-xs text-[#8b949e]">
            {resultCount === null
              ? '加载中…'
              : `${resultCount} / ${candidateCount ?? '?'} 只`}
          </span>
          <button
            onClick={onReset}
            className="text-xs px-2 py-1 rounded bg-[#21262d] hover:bg-[#30363d] border border-[#30363d]"
          >
            重置
          </button>
        </div>
      </div>

      {/* Existing fields (placeholder — Task 9 adds the new groups below) */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {/* T8 leaves these inputs as-is from Picks.tsx; T9 swaps them out. */}
        <div className="text-xs text-[#6e7681]">视图 / Top N / 窗口 / minTop 在 Task 9 添加</div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Stub Picks.tsx to use the new hook + FilterBar (placeholder UI)**

Replace the existing `frontend/src/pages/Picks.tsx` entirely with a slimmer scaffold that delegates filter UI to FilterBar. **Important**: we keep the existing visual table render intact; only the filter bar moves.

```typescript
import { Link } from 'react-router-dom';
import { useScreen } from '@/models/hooks';
import { cn } from '@/lib/utils';
import { FilterBar } from './picks/FilterBar';
import { useFilterParams } from './picks/useFilterParams';
import type { FilterParams } from './picks/types';

export default function Picks() {
  const [params, update, reset] = useFilterParams();

  const { data, isPending, error } = useScreen(toQueryParams(params));

  const filteredItems = data
    ? data.items.filter((it) => (it.consensus ?? 0) >= params.min_consensus)
    : [];

  return (
    <div className="space-y-6 max-w-6xl">
      <header>
        <h1 className="text-2xl font-semibold">选股工作台</h1>
        <p className="text-sm text-[#8b949e] mt-1">
          基于滚动重训集成模型的横截面打分排名 · 可按价格 / 涨跌幅 / 振幅 / 量比 / 板块 / ST 等筛选
        </p>
      </header>

      <FilterBar
        params={params}
        resultCount={data ? filteredItems.length : null}
        candidateCount={data ? data.items.length : null}
        onChange={update}
        onReset={reset}
      />

      {/* Results table */}
      <div className="rounded-lg border border-[#30363d] bg-[#0d1117] p-5">
        <h2 className="text-sm font-semibold text-[#8b949e] uppercase tracking-wider mb-3">
          结果 {data ? `(${filteredItems.length}/${data.items.length})` : ''}
        </h2>
        {error ? (
          <div className="text-red-400 text-sm">加载失败: {(error as Error).message}</div>
        ) : isPending ? (
          <div className="text-[#8b949e] text-sm">加载中…</div>
        ) : data && filteredItems.length === 0 ? (
          <div className="text-[#8b949e] text-sm">没有符合条件的股票 — 试试放宽 [最高单价] 或 [板块] 筛选。</div>
        ) : data ? (
          <ResultsTable items={filteredItems} />
        ) : null}
      </div>

      {data && (
        <div className="text-xs text-[#6e7681] grid grid-cols-2 md:grid-cols-4 gap-4">
          <div><span className="uppercase tracking-wider">experiment</span><div className="font-mono text-[#8b949e] mt-1">{data.experiment}</div></div>
          <div><span className="uppercase tracking-wider">recorder_id</span><div className="font-mono text-[#8b949e] mt-1 truncate">{data.recorder_id}</div></div>
          <div><span className="uppercase tracking-wider">latest_date</span><div className="font-mono text-[#8b949e] mt-1">{data.latest_date}</div></div>
          <div><span className="uppercase tracking-wider">universe_size</span><div className="font-mono text-[#8b949e] mt-1">{data.universe_size.toLocaleString()}</div></div>
        </div>
      )}
    </div>
  );
}

function toQueryParams(p: FilterParams): Parameters<typeof useScreen>[0] {
  return {
    top: p.top,
    days: p.days,
    min_top: p.min_top,
    view: p.view,
    min_price: p.min_price,
    max_price: p.max_price,
    pct_change_n: p.pct_change_n,
    min_pct_change: p.min_pct_change,
    max_pct_change: p.max_pct_change,
    min_amplitude: p.min_amplitude,
    max_amplitude: p.max_amplitude,
    min_vol_ratio: p.min_vol_ratio,
    max_vol_ratio: p.max_vol_ratio,
    new_high_n: p.new_high_n,
    boards: p.boards,
    exclude_st: p.exclude_st,
  };
}

function ResultsTable({ items }: { items: any[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-xs uppercase tracking-wider text-[#6e7681] border-b border-[#30363d]">
            <th className="py-2 pr-4">rank</th>
            <th className="py-2 pr-4">代码</th>
            <th className="py-2 pr-4">名称</th>
            <th className="py-2 pr-4 text-right">单价 ¥</th>
            <th className="py-2 pr-4 text-right">100 股 ¥</th>
            <th className="py-2 pr-4 text-right">涨跌5d</th>
            <th className="py-2 pr-4 text-right">振幅</th>
            <th className="py-2 pr-4 text-right">量比</th>
            <th className="py-2 pr-4">板块</th>
            <th className="py-2 pr-4 text-right">共识</th>
          </tr>
        </thead>
        <tbody>
          {items.map((item) => (
            <tr key={item.symbol} className="border-b border-[#21262d] hover:bg-[#161b22] transition cursor-pointer">
              <td className="py-2 pr-4 font-mono text-[#8b949e]">{item.rank}</td>
              <td className="py-2 pr-4">
                <Link to={`/charts/${item.symbol}`} className="font-mono text-[#58a6ff] hover:underline">{item.symbol}</Link>
              </td>
              <td className="py-2 pr-4">
                <Link to={`/charts/${item.symbol}`} className="hover:underline">{item.name}</Link>
              </td>
              <td className="py-2 pr-4 text-right font-mono text-[#e6edf3]">
                {item.last_price != null ? item.last_price.toFixed(2) : '—'}
              </td>
              <td className="py-2 pr-4 text-right font-mono text-[#8b949e]">
                {item.last_price != null ? '¥' + (item.last_price * 100).toLocaleString('zh-CN', { maximumFractionDigits: 0 }) : '—'}
              </td>
              <td className={cn('py-2 pr-4 text-right font-mono', pctColorClass(item.pct_change_5d))}>
                {item.pct_change_5d != null ? formatPct(item.pct_change_5d) : '—'}
              </td>
              <td className="py-2 pr-4 text-right font-mono">
                {item.amplitude != null ? (item.amplitude * 100).toFixed(2) + '%' : '—'}
              </td>
              <td className="py-2 pr-4 text-right font-mono">
                {item.vol_ratio != null ? item.vol_ratio.toFixed(2) : '—'}
              </td>
              <td className="py-2 pr-4 text-[#8b949e]">{labelBoard(item.board)}</td>
              <td className={cn('py-2 pr-4 text-right font-mono', consensusColorClass(item.consensus ?? 0))}>
                {(item.consensus ?? 0).toFixed(2)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function formatPct(v: number): string {
  return (v >= 0 ? '+' : '') + (v * 100).toFixed(2) + '%';
}

function pctColorClass(v: number | null | undefined): string {
  if (v == null) return 'text-[#8b949e]';
  if (v > 0.005) return 'text-green-400';
  if (v < -0.005) return 'text-red-400';
  return 'text-[#8b949e]';
}

function labelBoard(b: string | null | undefined): string {
  switch (b) {
    case 'main': return '主板';
    case 'gem': return '创业板';
    case 'star': return '科创板';
    case 'bj': return '北交所';
    case 'etf': return 'ETF';
    default: return '—';
  }
}

function consensusColorClass(v: number): string {
  if (v >= 0.78) return 'text-green-400';
  if (v >= 0.44) return 'text-yellow-400';
  return 'text-[#8b949e]';
}
```

- [ ] **Step 2.5: Verify FilterBar import path on Windows**

```
ls frontend/src/pages/picks/FilterBar.tsx frontend/src/pages/picks/useFilterParams.ts frontend/src/pages/picks/parse.ts frontend/src/pages/picks/types.ts
```

Expected: all 4 files exist.

- [ ] **Step 3: Typecheck**

```
cd frontend
npx tsc --noEmit
```

Expected: 0 errors.

- [ ] **Step 4: Manual smoke test in the running dev server**

Open `http://127.0.0.1:5173/picks`. Verify:
- The table still renders.
- The new 涨跌5d / 振幅 / 量比 / 板块 columns appear and populate.
- The FilterBar placeholder shows "视图 / Top N / 窗口 / minTop 在 Task 9 添加" + result count badge + 重置 button.

URL is empty (default filters). Try `http://127.0.0.1:5173/picks?max_price=15&boards=star` → after T9 the boards filter will be honoured; for now the backend already applies it via the URL state hook.

- [ ] **Step 5: Commit**

```
git add frontend/src/pages/picks/FilterBar.tsx frontend/src/pages/Picks.tsx
git commit -m "feat(screener): extract FilterBar component + URL-driven Picks page"
```

---

## Phase C — Frontend filter UI

### Task 9: Add Tier 1 filter inputs to FilterBar

**Files:**
- Modify: `frontend/src/pages/picks/FilterBar.tsx`

- [ ] **Step 1: Replace the placeholder body of FilterBar with the full Tier 1 UI**

`frontend/src/pages/picks/FilterBar.tsx`:

```typescript
import { useEffect, useState } from 'react';
import type { Board, FilterParams, NewHighN, PctChangeN, View } from './types';
import { BOARDS, NEW_HIGH_N_OPTIONS, PCT_CHANGE_N_OPTIONS } from './types';

interface FilterBarProps {
  params: FilterParams;
  resultCount: number | null;
  candidateCount: number | null;
  onChange: (patch: Partial<FilterParams>) => void;
  onReset: () => void;
}

const VIEW_OPTIONS: { value: View; label: string }[] = [
  { value: 'ensemble', label: '集成 (Ensemble)' },
  { value: 'lightgbm', label: 'LightGBM' },
  { value: 'alstm', label: 'ALSTM' },
  { value: 'tra', label: 'TRA' },
];

export function FilterBar({ params, resultCount, candidateCount, onChange, onReset }: FilterBarProps) {
  return (
    <div className="rounded-lg border border-[#30363d] bg-[#0d1117] p-5 space-y-5">
      {/* Header: result count + reset */}
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-[#8b949e] uppercase tracking-wider">筛选</h2>
        <div className="flex items-center gap-3">
          <span className="text-xs text-[#8b949e]">
            {resultCount === null ? '加载中…' : `${resultCount} / ${candidateCount ?? '?'} 只`}
          </span>
          <button
            onClick={onReset}
            className="text-xs px-2 py-1 rounded bg-[#21262d] hover:bg-[#30363d] border border-[#30363d]"
          >
            重置
          </button>
        </div>
      </div>

      {/* Group 1: 基础 (immediate update) */}
      <div>
        <h3 className="text-[10px] text-[#6e7681] uppercase tracking-wider mb-2">基础</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <Select label="视图" value={params.view} options={VIEW_OPTIONS} onChange={(v) => onChange({ view: v as View })} />
          <NumberField label="Top N" value={params.top} min={1} max={300} onChange={(v) => onChange({ top: v })} />
          <NumberField label="窗口天数" value={params.days} min={1} max={60} onChange={(v) => onChange({ days: v })} />
          <NumberField label="最少进 top N 天数" value={params.min_top} min={0} max={params.days} onChange={(v) => onChange({ min_top: v })} />
        </div>
      </div>

      {/* Group 2: 价格 (debounced) */}
      <div>
        <h3 className="text-[10px] text-[#6e7681] uppercase tracking-wider mb-2">价格 (¥/股)</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <DebouncedField label="最低单价" value={params.min_price} step={0.01} min={0} placeholder="无下限" onCommit={(v) => onChange({ min_price: v })} />
          <DebouncedField label="最高单价" value={params.max_price} step={0.01} min={0} placeholder="无上限" onCommit={(v) => onChange({ max_price: v })} />
        </div>
        <p className="text-[10px] text-[#6e7681] mt-1">
          A 股 100 股/手 · 4000 元买入 → 单价 ≤ ¥40 · ETF 可单股买
        </p>
      </div>

      {/* Group 3: 走势 (debounced) */}
      <div>
        <h3 className="text-[10px] text-[#6e7681] uppercase tracking-wider mb-2">走势</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
          <Select
            label="涨跌幅 N 日"
            value={String(params.pct_change_n)}
            options={PCT_CHANGE_N_OPTIONS.map((n) => ({ value: String(n), label: `${n} 日` }))}
            onChange={(v) => onChange({ pct_change_n: Number(v) as PctChangeN })}
          />
          <DebouncedField label="涨跌幅 min (%)" value={pctToUi(params.min_pct_change)} step={0.1} placeholder="不限" onCommit={(v) => onChange({ min_pct_change: pctFromUi(v) })} />
          <DebouncedField label="涨跌幅 max (%)" value={pctToUi(params.max_pct_change)} step={0.1} placeholder="不限" onCommit={(v) => onChange({ max_pct_change: pctFromUi(v) })} />
          <DebouncedField label="振幅 min (%)" value={pctToUi(params.min_amplitude)} step={0.1} min={0} placeholder="不限" onCommit={(v) => onChange({ min_amplitude: pctFromUi(v) })} />
          <DebouncedField label="振幅 max (%)" value={pctToUi(params.max_amplitude)} step={0.1} min={0} placeholder="不限" onCommit={(v) => onChange({ max_amplitude: pctFromUi(v) })} />
          <DebouncedField label="量比 min" value={params.min_vol_ratio} step={0.1} min={0} placeholder="不限" onCommit={(v) => onChange({ min_vol_ratio: v })} />
          <DebouncedField label="量比 max" value={params.max_vol_ratio} step={0.1} min={0} placeholder="不限" onCommit={(v) => onChange({ max_vol_ratio: v })} />
          <Select
            label="创 N 日新高"
            value={String(params.new_high_n)}
            options={NEW_HIGH_N_OPTIONS.map((n) => ({ value: String(n), label: n === 0 ? '关闭' : `${n} 日` }))}
            onChange={(v) => onChange({ new_high_n: Number(v) as NewHighN })}
          />
        </div>
      </div>

      {/* Group 4: 属性 (immediate) */}
      <div>
        <h3 className="text-[10px] text-[#6e7681] uppercase tracking-wider mb-2">属性</h3>
        <div className="flex flex-col md:flex-row md:items-center gap-4">
          <BoardsCheckboxes value={params.boards} onChange={(boards) => onChange({ boards })} />
          <label className="flex items-center gap-2 text-sm cursor-pointer">
            <input
              type="checkbox"
              checked={params.exclude_st}
              onChange={(e) => onChange({ exclude_st: e.target.checked })}
              className="accent-[#1f6feb]"
            />
            <span>排除 ST</span>
          </label>
        </div>
      </div>

      {/* Group 5: 共识 (immediate; UI-only filter, doesn't hit backend) */}
      <div>
        <h3 className="text-[10px] text-[#6e7681] uppercase tracking-wider mb-2">共识</h3>
        <label className="block max-w-md">
          <span className="text-xs text-[#6e7681]">最低共识 ({params.min_consensus.toFixed(2)})</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={params.min_consensus}
            onChange={(e) => onChange({ min_consensus: Number(e.target.value) })}
            className="mt-1 w-full h-9 accent-[#1f6feb]"
          />
        </label>
      </div>
    </div>
  );
}

// === sub-components ===

function NumberField({ label, value, min, max, onChange }: { label: string; value: number; min: number; max: number; onChange: (v: number) => void }) {
  return (
    <label className="block">
      <span className="text-xs text-[#6e7681] uppercase tracking-wider">{label}</span>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        onChange={(e) => onChange(Math.max(min, Math.min(max, Number(e.target.value) || min)))}
        className="mt-1 w-full rounded-md bg-[#161b22] border border-[#30363d] px-3 h-9 text-sm focus:outline-none focus:border-[#1f6feb]"
      />
    </label>
  );
}

function Select<T extends string>({ label, value, options, onChange }: { label: string; value: T; options: { value: T; label: string }[]; onChange: (v: T) => void }) {
  return (
    <label className="block">
      <span className="text-xs text-[#6e7681] uppercase tracking-wider">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
        className="mt-1 w-full rounded-md bg-[#161b22] border border-[#30363d] px-3 h-9 text-sm focus:outline-none focus:border-[#1f6feb]"
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>{opt.label}</option>
        ))}
      </select>
    </label>
  );
}

/** Numeric field that debounces user typing before committing to the parent.
 *  500ms debounce; commits empty string as null. */
function DebouncedField({
  label, value, step, min, placeholder, onCommit,
}: {
  label: string;
  value: number | null;
  step?: number;
  min?: number;
  placeholder?: string;
  onCommit: (v: number | null) => void;
}) {
  const [local, setLocal] = useState<string>(value === null ? '' : String(value));

  // External -> local: keep in sync if parent value changes via reset / URL
  useEffect(() => {
    setLocal(value === null ? '' : String(value));
  }, [value]);

  // Local -> external (debounced)
  useEffect(() => {
    const handle = setTimeout(() => {
      if (local === '') {
        if (value !== null) onCommit(null);
        return;
      }
      const parsed = Number(local);
      if (!Number.isFinite(parsed)) return;
      if (parsed !== value) onCommit(parsed);
    }, 500);
    return () => clearTimeout(handle);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [local]);

  return (
    <label className="block">
      <span className="text-xs text-[#6e7681] uppercase tracking-wider">{label}</span>
      <input
        type="number"
        value={local}
        step={step}
        min={min}
        placeholder={placeholder}
        onChange={(e) => setLocal(e.target.value)}
        className="mt-1 w-full rounded-md bg-[#161b22] border border-[#30363d] px-3 h-9 text-sm focus:outline-none focus:border-[#1f6feb]"
      />
    </label>
  );
}

function BoardsCheckboxes({ value, onChange }: { value: Board[]; onChange: (next: Board[]) => void }) {
  const toggle = (b: Board) => {
    const set = new Set(value);
    set.has(b) ? set.delete(b) : set.add(b);
    onChange(Array.from(set));
  };
  return (
    <fieldset>
      <legend className="text-xs text-[#6e7681] uppercase tracking-wider mb-1">板块 (多选 = 并集)</legend>
      <div className="flex flex-wrap gap-3">
        {BOARDS.map(({ value: b, label }) => (
          <label key={b} className="flex items-center gap-1 text-sm cursor-pointer">
            <input
              type="checkbox"
              checked={value.includes(b)}
              onChange={() => toggle(b)}
              className="accent-[#1f6feb]"
            />
            <span>{label}</span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}

// pct utility: backend uses raw decimal (0.05 = 5%), UI shows percent (5.0)
// e.g. pctToUi(0.0532) === 5.32, pctFromUi(5.32) === 0.0532.
function pctToUi(v: number | null): number | null {
  return v === null ? null : Math.round(v * 10000) / 100;
}

function pctFromUi(v: number | null): number | null {
  return v === null ? null : v / 100;
}
```

- [ ] **Step 2: Typecheck**

```
cd frontend
npx tsc --noEmit
```

Expected: 0 errors.

- [ ] **Step 3: Manual smoke test**

Open `http://127.0.0.1:5173/picks` (backend on :8000). Verify:
- 5 filter groups appear: 基础 / 价格 / 走势 / 属性 / 共识
- 改 "最高单价 25", 等约 500ms, 表格刷新, 所有结果 ≤ 25 元
- 勾 "排除 ST"（默认已勾）, 切换 → URL 同步
- 多选板块 (主板 + 创业板) → URL `?boards=main,gem`, 表格只显示这两个板块
- 涨跌幅 N 日选 20, min 输入 5 → 表格只显示 20 日累计涨幅 ≥ 5% 的票
- 创 60 日新高 → 表格只显示当日为 60 日新高的票
- 量比 min 1.5 → 表格只显示放量股
- 点 "重置" → 所有筛选回默认 (最高单价 = 30 / 排除 ST / 其他全空)
- URL 用浏览器 ← 后退按钮能回到上一组筛选

- [ ] **Step 4: Commit**

```
git add frontend/src/pages/picks/FilterBar.tsx
git commit -m "feat(screener): full Tier 1 filter UI with debounce + URL state"
```

---

### Task 10: Empty-state hint refinement + table indicator

**Files:**
- Modify: `frontend/src/pages/Picks.tsx`

- [ ] **Step 1: Make the empty-state message diagnostic**

Replace the existing empty-state block in `frontend/src/pages/Picks.tsx`:

```typescript
        ) : data && filteredItems.length === 0 ? (
          <EmptyState params={params} totalCandidates={data.items.length} />
        ) : data ? (
```

Add the component at the bottom of the file:

```typescript
function EmptyState({ params, totalCandidates }: { params: FilterParams; totalCandidates: number }) {
  const culprits: string[] = [];
  if (params.max_price !== null && params.max_price < 100) culprits.push('最高单价');
  if (params.boards.length > 0 && params.boards.length < BOARDS_COUNT) culprits.push('板块多选');
  if (params.new_high_n !== 0) culprits.push('创 N 日新高');
  if (params.min_pct_change !== null && params.min_pct_change > 0) culprits.push('涨跌幅 min');
  if (params.min_vol_ratio !== null && params.min_vol_ratio > 1) culprits.push('量比 min');
  if (params.min_consensus > 0.5) culprits.push('最低共识');

  return (
    <div className="text-sm text-[#8b949e]">
      <p>没有符合条件的股票 ({totalCandidates} 候选都被筛掉)。</p>
      {culprits.length > 0 && (
        <p className="mt-2">
          可能太严的筛选: <span className="text-yellow-400">{culprits.join(' · ')}</span>
        </p>
      )}
    </div>
  );
}

const BOARDS_COUNT = 5; // main / gem / star / bj / etf
```

Add the import for `FilterParams` and `BOARDS` at the top:

```typescript
import type { FilterParams } from './picks/types';
```

(BOARDS_COUNT is local; we don't need to import the full BOARDS list here.)

- [ ] **Step 2: Add a subtle loading indicator during refetch**

In the `<ResultsTable>` parent block, wrap the table with a fading overlay during refetch. Modify Picks.tsx's results block:

```typescript
        ) : data ? (
          <div className={cn('relative', isPending ? 'opacity-50 pointer-events-none' : '')}>
            <ResultsTable items={filteredItems} />
          </div>
        ) : null}
```

Wait — `isPending` is `false` after first load; we want a "fetching" indicator on refetch. Use `isFetching` from TanStack Query. Adjust the destructure:

```typescript
const { data, isPending, isFetching, error } = useScreen(toQueryParams(params));
```

And:

```typescript
        ) : data ? (
          <div className={cn('relative', isFetching ? 'opacity-60 transition-opacity' : 'transition-opacity')}>
            <ResultsTable items={filteredItems} />
          </div>
        ) : null}
```

- [ ] **Step 3: Typecheck**

```
cd frontend
npx tsc --noEmit
```

Expected: 0 errors.

- [ ] **Step 4: Manual smoke test**

- Set 最高单价 = 1 (no stocks qualify) → empty state shows "可能太严的筛选: 最高单价"
- Set 板块 = 北交所 only (no BJ predictions in cn_data_bs CSI300/500 universe) → empty state shows "板块多选"
- Change any filter while data is showing → table dims briefly during fetch

- [ ] **Step 5: Commit**

```
git add frontend/src/pages/Picks.tsx
git commit -m "feat(screener): diagnostic empty-state + refetch-fade indicator"
```

---

## Phase D — Verification

### Task 11: End-to-end verification + acceptance checklist

**Files:**
- None (verification only)

- [ ] **Step 1: Run backend test suite**

```
cd backend
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest -v
```

Expected: green (the 3 new test files plus existing). Allowed: skips on data-less integration tests.

- [ ] **Step 2: Run frontend typecheck + build**

```
cd frontend
npx tsc --noEmit
npm run build
```

Expected: 0 TS errors, build succeeds.

- [ ] **Step 3: Manual acceptance checklist (UI)**

Walk through each item, ticking it off as you confirm in the browser:

- [ ] Open `/picks` — default URL is `/picks` (no query string), default filters applied (max_price=30, exclude_st=true).
- [ ] Result count badge (`X / Y 只`) updates within ~500ms of any filter change.
- [ ] Add `?max_price=15&boards=main,gem` to the URL → page loads with those filters pre-applied.
- [ ] Click "重置" → URL clears back to `/picks`, all filters return to defaults.
- [ ] Multi-select boards (主板 + 创业板) → table contains only entries with `board ∈ {main, gem}`.
- [ ] Set 涨跌幅 N 日 = 20, min = 5 → all rows have `pct_change_5d >= 5%` displayed in the column.
- [ ] Set 创新高 N = 60 → all rows are actual 60-day new highs.
- [ ] Toggle 排除 ST off → ST entries appear in the table (verify by name).
- [ ] Drag 共识 slider to 0.7 → fewer rows; all rows have `consensus ≥ 0.70`.
- [ ] Browser back button → reverts to previous filter URL.
- [ ] Filter producing 0 results → empty-state names the likely culprit.
- [ ] Table column 涨跌5d colored green/red based on sign.

- [ ] **Step 4: Commit a small "tier 1 acceptance" note**

```
# No code commit needed. If documentation update is desired, edit CHANGELOG or docs.
```

(Skip if no doc update; this task is a checkpoint, not a commit.)

---

## Known Followups (Tier 2 / Tier 3 / Tier 4)

These are explicitly **not** part of this plan; the research report §3 maps them out for future plans:

1. **Tier 2** — Industry classification + market cap + 次新股: needs a one-time `production/fetch_stock_meta.py` script that hits baostock's stock-basic-info + industry tables and writes a parquet cached monthly.
2. **Tier 3** — PE / PB / ROE / 净利润同比: needs baostock quarterly financial reports; cached after each earnings season.
3. **Tier 4** — 主力净流入 / 北向 / 龙虎榜: needs akshare daily feeds + a new background job.

---

## Acceptance Criteria

Tier 1 is done when:

- All 11 tasks committed, each with green TDD cycle.
- `/api/models/screen` honours every Tier 1 query param with AND semantics; out-of-range enum values return 400.
- The Picks page renders the 5-group filter bar; each control is bound to URL state and to backend params with the right debounce class (immediate vs 500ms).
- `npx tsc --noEmit` is clean.
- Manual checklist (Task 11 step 3) all green on the user's machine.

---

**End of plan.**
