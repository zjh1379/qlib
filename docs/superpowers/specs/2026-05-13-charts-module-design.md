# Charts Module + P1 Backend Bootstrap — Design

**Status**: Approved (2026-05-13)
**Scope**: Finish the `charts/` module so `backend/app/charts/router.py` and its tests work end-to-end, including the missing P1 backend bootstrap (`core/`, `main.py`, `pyproject.toml`) that the already-committed `ops/` module also depends on.

## Goal

Make the P1 backend runnable and the chart endpoint useful:

- `GET /api/charts/{symbol}?start=&end=&with_pred=&experiment=` returns real OHLCV from qlib plus an optional prediction series resolved by qlib recorder ID.
- The existing `backend/app/charts/tests/test_router.py` passes against real qlib data.
- The already-committed `/api/ops/health` endpoint actually runs (currently it can't — its `app.core.qlib_adapter` import fails).

## Non-goals

- Training models, running qlib workflows, or persisting experiments.
- DB / Alembic / settings UI / job control (deferred P2+).
- Production deployment, auth, rate limiting.
- Frontend changes.

## Decisions log

| # | Decision | Choice |
|---|---|---|
| 1 | Scope | Charts + full P1 backend bootstrap (`core/`, `main.py`, `pyproject.toml`) |
| 2 | `predicted` source | Real qlib experiment lookup |
| 3 | Experiment resolution | Caller passes recorder ID directly |
| 4 | Row shape | Full OHLCV; `predicted` parallel list of `{date, value}` |
| 5 | Sync-in-async | `asyncio.to_thread(...)` wrappers around all qlib calls |
| 6 | `symbol_missing` vs `ohlcv_empty` | Calendar-based disambiguation (B2) |

## File plan

New files under `backend/`:

```
pyproject.toml
app/__init__.py
app/main.py
app/core/__init__.py
app/core/config.py
app/core/exceptions.py
app/core/qlib_adapter.py
app/charts/__init__.py
app/charts/schemas.py
app/charts/service.py
app/charts/tests/__init__.py
app/charts/tests/test_service.py        # new unit tests with mocked adapter
app/core/tests/__init__.py
app/core/tests/test_exceptions.py
```

Already present, unchanged: `app/charts/router.py`, `app/charts/tests/test_router.py`, `app/ops/*`.

**Module boundaries**: `core/qlib_adapter` is the *only* file that imports `qlib.*` or `mlflow.*`. Service stays adapter-typed (`list[dict]`) so unit tests monkeypatch the adapter without needing qlib installed.

## Components

### `app/core/exceptions.py`

```python
class BusinessError(Exception):
    def __init__(self, code: str, message: str | None = None, http_status: int = 400):
        self.code = code
        self.message = message or code
        self.http_status = http_status
        super().__init__(self.message)

    def as_response_dict(self) -> dict:
        return {"code": self.code, "message": self.message}
```

Codes raised by charts:

| Code | HTTP | When |
|---|---|---|
| `bad_date_range` | 400 | `start > end`, or either date unparseable |
| `symbol_missing` | 404 | Window contains trading days, but symbol returned zero OHLCV rows |
| `ohlcv_empty` | 404 | Window contains zero trading days per qlib calendar (e.g. all-weekend) |
| `recorder_missing` | 404 | `with_pred=true`, `experiment` given, but recorder ID not found |

`with_pred=true` with `experiment=None` does NOT error — predictions are silently empty (200).

### `app/core/config.py`

```python
class Settings(BaseSettings):
    qlib_provider_uri: str = "~/.qlib/qlib_data/cn_data"
    qlib_region: str = "cn"
    mlflow_tracking_uri: str | None = None
    model_config = SettingsConfigDict(env_prefix="QLIB_COMPANION_", env_file=".env")

_settings = Settings()
def get_settings() -> Settings: return _settings
```

`mlflow_tracking_uri=None` lets qlib use its own default. Env vars are namespaced `QLIB_COMPANION_*` so they don't collide with qlib's own envs.

### `app/core/qlib_adapter.py`

Owns all `qlib.*` and `mlflow.*` imports.

```python
def init_qlib_once() -> None: ...
def get_calendar_end() -> datetime.date: ...
def get_calendar_in_range(start: str, end: str) -> list[datetime.date]: ...
def fetch_ohlcv(symbol: str, start: str, end: str) -> list[dict]: ...
def fetch_prediction(recorder_id: str, symbol: str, start: str, end: str) -> list[dict]: ...
```

- `init_qlib_once` is idempotent via a module-level flag; reads provider URI + region from `get_settings()`.
- `fetch_ohlcv` uses `D.features([symbol], ["$open","$high","$low","$close","$volume"], start, end, freq="day")` and converts to `[{"date","open","high","low","close","volume"}]`.
- `fetch_prediction` resolves the recorder by ID. Since `qlib.workflow.R.get_recorder(recorder_id=...)` requires an experiment context, the adapter iterates `R.list_experiments()` and looks up the recorder in each until found. Raises `BusinessError("recorder_missing", http_status=404)` on miss. Loads `pred.pkl` artifact, filters MultiIndex `(datetime, instrument)` by symbol+window, returns `[{"date","value"}]`.
- Possible future swap: `mlflow.tracking.MlflowClient.get_run(recorder_id)` direct, if the iteration proves slow. Same semantics.

### `app/charts/schemas.py`

```python
class OhlcvRow(BaseModel):
    date: str    # ISO YYYY-MM-DD
    open: float; high: float; low: float; close: float; volume: float

class PredictedPoint(BaseModel):
    date: str
    value: float

class ChartPayload(BaseModel):
    symbol: str
    actual: list[OhlcvRow]
    predicted: list[PredictedPoint]
    experiment: str | None = None   # echoed only when actually used
```

`date` is an ISO string (not `date`) so JSON is TZ-stable for the frontend. `predicted` is parallel to `actual`, not merged.

### `app/charts/service.py`

```python
async def get_chart(symbol, start, end, with_pred, experiment) -> ChartPayload:
    _validate_dates(start, end)                             # bad_date_range

    await asyncio.to_thread(init_qlib_once)
    actual = await asyncio.to_thread(fetch_ohlcv, symbol, start, end)

    if not actual:
        cal = await asyncio.to_thread(get_calendar_in_range, start, end)
        raise BusinessError("symbol_missing" if cal else "ohlcv_empty", http_status=404)

    predicted: list[dict] = []
    if with_pred and experiment:
        predicted = await asyncio.to_thread(fetch_prediction, experiment, symbol, start, end)

    return ChartPayload(
        symbol=symbol,
        actual=[OhlcvRow(**r) for r in actual],
        predicted=[PredictedPoint(**p) for p in predicted],
        experiment=experiment if (with_pred and experiment) else None,
    )
```

Validation happens before qlib init so bad input doesn't pay init cost.

### `app/main.py`

```python
app = FastAPI(title="Qlib Companion", version="0.1.0")

@app.exception_handler(BusinessError)
async def biz_handler(_, exc: BusinessError):
    return JSONResponse(status_code=exc.http_status, content=exc.as_response_dict())

app.include_router(ops_router, prefix="/api/ops", tags=["ops"])
app.include_router(charts_router, prefix="/api/charts", tags=["charts"])
```

### `backend/pyproject.toml`

```toml
[project]
name = "qlib-companion-backend"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
  "fastapi>=0.110", "uvicorn[standard]>=0.27",
  "pydantic>=2.6", "pydantic-settings>=2.2",
  "pyqlib", "pandas", "mlflow",
]
[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio", "httpx"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["app"]
pythonpath = ["."]
```

## Tests

### Existing (unchanged)
- `app/charts/tests/test_router.py` — integration smoke against real qlib data. Two cases: 200 happy path for SH600519, 404 for SH999999.
- `app/ops/tests/test_router.py` — health endpoint.

### New (added in this spec)

**`app/charts/tests/test_service.py`** — unit tests with `qlib_adapter` monkeypatched:

| Test | Setup | Asserts |
|---|---|---|
| `test_happy_path` | adapter returns 5 OHLCV rows, 3 pred rows | payload shape, `experiment` echoed |
| `test_symbol_missing` | adapter: `fetch_ohlcv=[]`, `get_calendar_in_range=[date(...)]` | raises `BusinessError` with `code="symbol_missing"`, `http_status=404` |
| `test_ohlcv_empty` | adapter: `fetch_ohlcv=[]`, `get_calendar_in_range=[]` | raises `BusinessError` with `code="ohlcv_empty"`, `http_status=404` |
| `test_bad_date_range` | start > end | raises `BusinessError` with `code="bad_date_range"`, `http_status=400` |
| `test_with_pred_no_experiment` | `with_pred=True, experiment=None` | returns 200, `predicted=[]`, `experiment=None` |
| `test_recorder_missing` | `fetch_prediction` raises `recorder_missing` | exception propagates unchanged |

**`app/core/tests/test_exceptions.py`** — single test of `as_response_dict()` shape.

No focused adapter tests at this milestone — integration coverage via the live `test_router.py` is enough; we add them when prediction wiring sees real use.

## Open risks / things to watch

1. **Recorder iteration cost.** `R.list_experiments()` + per-exp lookup is fine at small N but degrades. If it becomes user-visible, switch to direct `MlflowClient.get_run(recorder_id)`. Same semantics, no qlib indirection.
2. **`pred.pkl` artifact shape.** Qlib stores predictions as a `pandas.Series` indexed by `(datetime, instrument)` — but some workflows store DataFrames with a single `score` column. The adapter normalizes both shapes; if a third format shows up we extend in `fetch_prediction`.
3. **Symbol normalization.** The router accepts the symbol verbatim. qlib's CN data uses `SH600519`/`SZ000001` casing. We do not lowercase or reformat — caller responsibility. If frontend ends up sending lowercase, we add normalization here.
