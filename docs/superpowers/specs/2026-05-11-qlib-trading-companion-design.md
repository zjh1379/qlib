# Qlib Trading Companion — Design Spec

**Date**: 2026-05-11
**Status**: Approved for implementation planning
**Owner**: zhu.jinghu@northeastern.edu

---

## 1. Context

A working qlib-based prototype now exists in `production/`:

- `fetch_csi300_bs.py` — pulls baostock OHLCV for CSI300 stocks
- `add_benchmark.py` — adds SH000300 + writes `instruments/csi300.txt`
- `retrain_predict_cn_fresh.yaml` — LightGBM + Alpha158 training config
- `extract_predictions.py` — daily picks (SCREEN + WATCHLIST modes)
- `holdings_curve.py` — per-watchlist trajectory + verdict
- `etf_dashboard.py` — ETF composite signal (constituent-weighted)
- `update_qlib_data.ps1` — daily baostock refresh + dump_bin pipeline
- Scheduled task `QlibDailyUpdate` keeps `cn_data_bs/` current

These prove the data + ML pipeline works (latest run: IC 0.024, RIC 0.013, IR with cost 1.96 over 2025-01 to 2026-04 backtest). The user has confirmed prototype validation complete.

**Goal**: graduate from CLI prototype to a **trading companion app** the user can run on Windows, access from any of his devices via Tailscale, and (after wrapping) ship as a desktop + mobile app — primarily for buy/sell decision support, secondarily as a foundation for potential commercialization.

Core experience: chart-grade interactive K-line with model prediction overlay (IBKR-level quality), surrounded by signal/portfolio/ops panels.

## 2. Non-goals (v1)

To keep scope contained:

- Intraday/tick-level data (qlib daily only, per earlier decision)
- Real broker integration (manual transaction entry instead)
- Walk-forward rolling retrain (`OnlineManager`) — v2
- User accounts / multi-tenancy (Tailscale handles access)
- Public-internet exposure (no Cloudflare Tunnel / VPS deployment)
- Markets beyond CN A-share CSI300 (US extension is v1.x)
- Push notifications / wechat / email alerts — v1.1
- WebSocket / real-time push (HTTP polling sufficient at daily cadence)

## 3. Decisions Locked In Through Brainstorming

| # | Question | Choice | Implications |
|---|---|---|---|
| Q1 | Platform strategy | **API + web first, then progressive Tauri/Capacitor wrappers** | One React codebase, multi-target packaging. No multi-platform native codebases. |
| Q2 | MVP feature scope | **Full set**: charts + signals + holdings/P&L + data/model ops UI | ~2-3 weeks core implementation. All four domains shipped together. |
| Q3 | Remote access | **LAN + Tailscale** | No app-level auth in v1. Tailscale identity = access identity. |
| Q4 | Buy/sell record keeping | **App maintains its own ledger** | `transactions` table is source of truth; holdings derived. Manual entry; broker integration is v1.x. |
| Q5 | Model retrain cadence | **Weekly automatic + manual override** | Sun 22:00 cron via Task Scheduler; "retrain now" button always available. |
| Arch | Code organization | **Modular monolith (β)** | 5 domain modules; SQLAlchemy 2.x + Alembic; React + TanStack Query; ~3 weeks. |

## 3a. Pinned versions and conventions

- **Python**: 3.10 (matches existing `F:\Tools\Anaconda\envs\qlib` env, do not bump in v1).
- **Node**: 20.x LTS.
- **Time zone**: all server-side timestamps stored as UTC; all displayed timestamps and cron triggers in **Asia/Shanghai**. Task Scheduler runs in local time.
- **Trading calendar source of truth**: `~/.qlib/qlib_data/cn_data_bs/calendars/day.txt`. The backend treats this as authoritative for "is today a trading day" and forecast-shift math.
- **Data freshness SLA**: data is "fresh" if `calendar_end >= last_cn_trading_day`. UI shows a yellow banner if stale by 1 trading day, red if ≥ 2.

## 4. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Windows host (Tailscale node)                                  │
│                                                                 │
│  ┌────────────────────────────────────┐                         │
│  │  FastAPI single process (port 8000)│                         │
│  │                                    │                         │
│  │  modules: charts / data /          │   ┌──────────────────┐  │
│  │           portfolio / models / ops │◄─►│ qlib_data/       │  │
│  │                                    │   │   cn_data_bs/    │  │
│  │  core: db / qlib_adapter /         │   │     .bin files   │  │
│  │        config / logging /          │   └──────────────────┘  │
│  │        exceptions                  │                         │
│  │                                    │   ┌──────────────────┐  │
│  │  Pydantic schemas (boundary)       │◄─►│ mlruns/          │  │
│  │  SQLAlchemy ORM (interior)         │   │   experiments    │  │
│  │  Alembic migrations                │   │   pred.pkl       │  │
│  │  BackgroundTasks for async         │   └──────────────────┘  │
│  └────────────────────────────────────┘                         │
│         ▲                                  ┌──────────────────┐ │
│         │ REST + static                    │ app.db (SQLite)  │ │
│         ▼                                  │   watchlists     │ │
│  ┌────────────────────────────────────┐    │   transactions   │ │
│  │  React SPA (built with Vite,       │◄──►│   app_settings   │ │
│  │  served by FastAPI as static)      │    │   jobs           │ │
│  │                                    │    └──────────────────┘ │
│  │  TanStack Query / Zustand          │                         │
│  │  TradingView Lightweight Charts    │                         │
│  │  Tailwind + shadcn/ui              │                         │
│  └────────────────────────────────────┘                         │
└──────────────────────────────────┬──────────────────────────────┘
                                   │ Tailscale (WireGuard)
                ┌──────────────────┼──────────────────┐
                ▼                  ▼                  ▼
        Owner laptop        Owner phone (PWA)   Family device
        (browser)                               (browser)
```

### Stack

| Layer | Choice | Why not the alternative |
|---|---|---|
| Backend framework | FastAPI | Flask/Django lack native async + Pydantic + OpenAPI |
| ORM | SQLAlchemy 2.x async | raw sqlite3 means painful migrations later |
| Migrations | Alembic | Hand-written SQL breaks under team / branching |
| State DB | SQLite | Postgres overkill for one user; DuckDB is OLAP-leaning |
| Frontend | React + TypeScript | Vue/Svelte ecosystems smaller for shadcn-style libraries |
| Bundler | Vite | webpack slower |
| Server state | TanStack Query | Redux/Zustand poor at request caching |
| UI lib | Tailwind + shadcn/ui | MUI hard to restyle; Ant Design overweight |
| Charts | TradingView Lightweight Charts | ECharts second-best; Charting Library needs registration |
| Desktop wrap | Tauri (v1.1) | Electron 100+ MB; native Win desktop is 3x effort |
| Mobile wrap | Capacitor (v1.2) | React Native splits codebase |
| Async exec | FastAPI BackgroundTasks + Win Task Scheduler | Celery+Redis overkill for ~1 task/week |

### Three deliberate simplifications

1. **No Redis / queue**: single user; longest task is 7-min weekly retrain. FastAPI BackgroundTasks suffices.
2. **No WebSocket in v1**: pre-frozen by daily cadence. HTTP polling fine.
3. **Demo HTML from brainstorming is NOT reused**: `chart-overlay-v2.html`, `chart-final*.html` etc. were proof-of-concept under `.superpowers/brainstorm/`. Production chart code is implemented fresh in `frontend/src/charts/`.

## 5. Backend Module Boundaries

```
backend/
├── app/
│   ├── main.py                  # FastAPI app, mounts routers, lifecycle hooks
│   ├── cli.py                   # entry for Task Scheduler (e.g. `python -m app.cli refresh_data`)
│   │
│   ├── core/                    # shared kernel
│   │   ├── db.py                # AsyncSession factory
│   │   ├── qlib_adapter.py      # the ONLY module that imports qlib
│   │   ├── config.py            # pydantic-settings (paths, cron times, ports)
│   │   ├── logging.py           # structlog setup
│   │   └── exceptions.py        # base exception classes (BusinessError, etc.)
│   │
│   ├── charts/                  # M1: OHLCV + prediction overlay
│   ├── data/                    # M2: data freshness, refresh trigger
│   ├── portfolio/               # M3: holdings, transactions, watchlists, P&L
│   ├── models/                  # M4: mlflow experiments, screen, retrain
│   └── ops/                     # M5: settings, schedule, job logs, health
│
├── alembic/                     # database migrations
├── pyproject.toml
└── tests/integration/
```

Each module follows: `router.py` (FastAPI), `service.py` (business logic), `schemas.py` (Pydantic), optional `models.py` (SQLAlchemy ORM), optional `tasks.py` (background jobs), and `tests/`.

### Public contracts (the ONLY entry points each module exposes)

| Module | Public functions in `service.py` |
|---|---|
| **charts** | `get_chart(symbol, start, end, with_pred) → ChartPayload` |
| **data** | `refresh_data(force) → JobId`; `get_data_status() → DataStatus` |
| **portfolio** | `add_transaction(...)`, `get_holdings()`, `get_pnl(window)`, `list_watchlists()`, `add_watchlist_item(...)` |
| **models** | `trigger_retrain() → JobId`, `list_experiments()`, `get_latest_pred(symbol, days)`, `screen(filters)`, `etf_composite(etf_code, days)` |
| **ops** | `get_setting(key)`, `set_setting(key, value)`, `get_schedule_status()`, `list_jobs(kind, limit)` |

### Inter-module rules

- Always call **other modules' `service.py` functions**, never their routers or ORM models.
- `portfolio → models` is allowed (holdings need scores). `models → portfolio` is forbidden (model logic must not know about user positions).
- `charts` is an aggregator: it composes `data` and `models` outputs into a chart payload. It owns no tables.
- Communication uses Pydantic schemas; ORM objects never cross module boundaries.

## 6. Data Model

Three storage locations, each with one job:

| Store | Owns | Format |
|---|---|---|
| `app.db` (SQLite) | User-modifiable state | Relational (5 tables + 1 view) |
| `~/.qlib/qlib_data/cn_data_bs/` | OHLCV + calendar + instruments | qlib binary (.bin) |
| `mlruns/` | Experiments + pred.pkl + metrics | mlflow file store |

### Tables

```sql
-- Watchlist groups (holding / watch / screen)
CREATE TABLE watchlists (
    id           INTEGER PRIMARY KEY,
    name         TEXT NOT NULL,
    kind         TEXT NOT NULL CHECK (kind IN ('holding','watch','screen')),
    description  TEXT,
    sort_order   INTEGER DEFAULT 0,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE watchlist_items (
    id            INTEGER PRIMARY KEY,
    watchlist_id  INTEGER NOT NULL REFERENCES watchlists(id) ON DELETE CASCADE,
    symbol        TEXT NOT NULL,
    added_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes         TEXT,
    UNIQUE (watchlist_id, symbol)
);

CREATE TABLE transactions (
    id            INTEGER PRIMARY KEY,
    symbol        TEXT NOT NULL,
    kind          TEXT NOT NULL CHECK (kind IN ('buy','sell')),
    qty           REAL NOT NULL,
    price         REAL NOT NULL,
    fee           REAL DEFAULT 0,
    executed_at   TIMESTAMP NOT NULL,
    broker        TEXT,
    notes         TEXT,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_tx_symbol_time ON transactions(symbol, executed_at);

CREATE TABLE app_settings (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,                    -- JSON
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE jobs (
    id            INTEGER PRIMARY KEY,
    kind          TEXT NOT NULL CHECK (kind IN ('data_refresh','retrain','manual')),
    status        TEXT NOT NULL CHECK (status IN ('queued','running','succeeded','failed')),
    started_at    TIMESTAMP,
    finished_at   TIMESTAMP,
    log_path      TEXT,
    error         TEXT,
    extra         TEXT                            -- JSON: e.g. recorder_id, instruments_count
);
CREATE INDEX idx_jobs_kind_started ON jobs(kind, started_at DESC);
```

### Derived: current holdings (computed in `portfolio/service.py`)

```sql
WITH net AS (
  SELECT symbol,
         SUM(CASE WHEN kind='buy'  THEN qty ELSE 0 END) AS bought,
         SUM(CASE WHEN kind='sell' THEN qty ELSE 0 END) AS sold,
         SUM(CASE WHEN kind='buy'  THEN qty*price + fee ELSE 0 END) AS cost_in,
         SUM(CASE WHEN kind='sell' THEN qty*price - fee ELSE 0 END) AS cash_out
  FROM transactions GROUP BY symbol
)
SELECT
    symbol,
    bought - sold                          AS qty,
    cost_in / NULLIF(bought, 0)            AS avg_buy_price,
    (cost_in - cash_out) / NULLIF(bought - sold, 0) AS effective_cost
FROM net
WHERE bought - sold > 0;
```

The service layer combines this with latest close from `qlib_adapter.get_ohlcv()` to compute mark-to-market value and unrealized P&L.

### qlib_adapter public API (the only qlib touchpoint)

```python
def init_qlib_once() -> None
def get_ohlcv(symbols: list[str], start: str, end: str) -> pd.DataFrame
def get_calendar_end() -> date
def get_latest_recorder(experiment: str) -> Recorder
def load_pred(recorder_id: str) -> pd.Series   # multi-indexed by (date, instrument)
def get_csi300_instruments() -> list[str]
```

## 7. REST API

```
GET    /api/charts/{symbol}?start=&end=&with_pred=true
GET    /api/charts/{symbol}/forecast
GET    /api/charts/compare?symbols=A,B,C                    [v1.x]

GET    /api/data/status
POST   /api/data/refresh
GET    /api/data/jobs?kind=data_refresh&limit=10

GET    /api/portfolio/holdings
GET    /api/portfolio/pnl?window=30d
GET    /api/portfolio/transactions?symbol=&from=&to=
POST   /api/portfolio/transactions
PATCH  /api/portfolio/transactions/{id}
DELETE /api/portfolio/transactions/{id}
GET    /api/portfolio/watchlists
POST   /api/portfolio/watchlists
POST   /api/portfolio/watchlists/{id}/items
DELETE /api/portfolio/watchlists/{id}/items/{symbol}

GET    /api/models/experiments
GET    /api/models/experiments/{name}/latest
GET    /api/models/predictions/{symbol}?days=60
GET    /api/models/screen?top=&days=&min_top=&watchlist_id=
GET    /api/models/etf/{etf_code}/composite?days=60
POST   /api/models/retrain
GET    /api/models/jobs?kind=retrain&limit=10

GET    /api/ops/settings
PUT    /api/ops/settings/{key}
GET    /api/ops/schedule
GET    /api/ops/jobs/{id}/log
GET    /api/ops/health
```

Conventions:
- All POST/PATCH/PUT validated by Pydantic; errors return 4xx with `{"detail": "..."}`.
- All GET responses cacheable; TanStack Query handles staleness on the frontend.
- Long-running operations (`/refresh`, `/retrain`) return immediately with `job_id`; frontend polls `/api/ops/jobs/{id}/log` and `/api/models/jobs?...`.
- Type safety: backend Pydantic schemas are exported via `openapi-typescript` to generate frontend TypeScript types.

## 8. Frontend Pages

```
/                              Dashboard
                                   - Holdings P&L summary card
                                   - Top 5 picks for today
                                   - Watchlist strong signals (days_in_top ≥ 3)
                                   - Data freshness + last retrain time
                                   - System status indicators

/charts/:symbol                Single-ticker deep dive
                                   - Main: dual-layer candles (actual + predicted)
                                     with opacity toggles, time-window controls,
                                     forward forecast bars
                                   - Sub-pane: volume + score line
                                   - Side: model verdict + 5-day score table
                                   - Bottom: this stock's transaction history + add button

/charts/etf/:code              ETF page (same layout + constituent overlay)

/picks                         Screening workbench
                                   - Filters: top N, days, min_days_in_top, industry
                                   - Sortable table
                                   - Row actions: add to watchlist, open chart

/portfolio                     Holdings overview (grid of position cards)
/portfolio/transactions        Transaction history (sortable, filterable, exportable)
/portfolio/pnl                 Portfolio value curve vs CSI300 benchmark

/ops                           Ops home (status dashboard)
/ops/data                      Data source status + manual refresh + job history
/ops/models                    mlflow experiment comparison + manual retrain
                                  + default recorder selector
/ops/settings                  Cron times, thresholds, defaults
/ops/jobs/:id                  Live job log viewer
```

### State management

| Layer | Tool | Use for |
|---|---|---|
| Server cache | TanStack Query | All `/api/*` data, automatic refetch on stale + post-mutation invalidate |
| UI ephemeral state | Zustand | Selected stock, modal flags, chart toolbar |

Each module has a typed `hooks.ts` file with `useChart()`, `useHoldings()`, `useScreen()`, etc., backed by the openapi-typescript types.

## 9. Background Tasks, Errors, Testing, Deployment

### Scheduling

```
Windows Task Scheduler                    FastAPI BackgroundTasks
─────────────────────                    ───────────────────────
17:30 weekdays:  data_refresh             User clicks "refresh now" or "retrain now"
22:00 Sundays:   retrain                  → in-process async task
Both call `python -m app.cli ...`        → write to jobs table + logs/
which uses the same service functions    → frontend polls job status
```

CLI entry and BackgroundTask share the **same service code**; never write the logic twice.

### Error model

```
service.py raises BusinessError(code, detail, context)
        ↓
FastAPI exception handler middleware
        ↓
JSON: { "detail": "...", "code": "...", "context": {...} }
        ↓
TanStack Query onError → toast / inline banner
```

Four error classes:

| Class | HTTP | Example | Handling |
|---|---|---|---|
| User input invalid | 400 | Delete a non-existent transaction | Friendly inline message |
| State conflict | 409 | Trigger screen before data refreshed | Banner with "refresh first" CTA |
| External dependency | 503 | baostock 429, qlib I/O failure | Toast + job status `failed` + logged |
| Uncaught | 500 | Bug | Stack trace logged to `logs/app-{date}.log`; Sentry hookup is v1.1 |

Structured logs via `structlog`; Ops module surfaces last 200 lines of `logs/app-{date}.log`.

### Testing pyramid

```
backend/
├── app/{module}/tests/      70% unit: service functions, mock qlib_adapter + db
├── tests/integration/       20% integration: TestClient + temp SQLite + fake pred.pkl
└── tests/contract/          10% contract: OpenAPI schema no breaking change

frontend/
├── *.test.tsx               Vitest + RTL for components
└── e2e/                     Playwright (v1.1) for smoke flows
```

**Non-negotiable v1 tests**:

1. P&L computation (realized + unrealized) — wrong formula breaks every dollar figure.
2. Transaction insert → derived holdings view stays consistent.
3. qlib_adapter init failure → graceful 503 + UI banner, never a crash page.

CI: GitHub Actions on private repo, runs full test suite + ruff + mypy + tsc on each PR.

### Deployment

```
Step 1: Local Windows service
  - Python 3.10 in F:\Tools\Anaconda\envs\qlib (existing env)
  - NSSM wraps `uvicorn app.main:app --host 0.0.0.0 --port 8000`
    → Windows service `QlibCompanion`, auto-start on boot
  - Vite build output served by FastAPI as static (single binary feel)

Step 2: Tailscale
  - Install Tailscale on the Windows host
  - Open Windows Firewall for port 8000 inbound, scope: Tailscale only
  - Access from any signed-in device: http://qlib-pc.tailnet-xxxx.ts.net:8000

Step 3: Desktop wrap (v1.1)
  - Tauri 1.x wraps the React build into .msi installer
  - Launches webview pointed at Tailscale host (or localhost when local)

Step 4: Mobile wrap (v1.2)
  - Capacitor wraps the same React build into .ipa / .apk
  - Still connects to home backend via Tailscale
```

### Authentication

**v1 has no app-level login.** Tailscale's network identity gates access. When sharing beyond the personal Tailnet (friends trialing, commercial users), add OAuth at routes — that's a v1.x change to the FastAPI middleware, no schema/codebase rewrite.

## 10. Acceptance Criteria — "v1 done" looks like

Functional:

- [ ] User opens app on phone over Tailscale, sees today's P&L + top picks within 3 seconds.
- [ ] User taps a stock, sees interactive chart with overlay + forecast bars.
- [ ] User adds a buy transaction via UI; holdings page updates immediately.
- [ ] User clicks "retrain now"; sees job progress; new pred.pkl is used next page load.
- [ ] User clicks "refresh data"; baostock pull + dump_bin + add_benchmark runs in background.
- [ ] On Sunday 22:00, retrain runs automatically; on weekdays 17:30, data refresh runs.

Quality:

- [ ] All P&L numbers reconcile to manual spreadsheet within 1 cent.
- [ ] Test coverage ≥ 70% on `service.py` files.
- [ ] No `print()` or `console.log()` in non-test code.
- [ ] OpenAPI spec at `/docs` is current.
- [ ] Backend boots from cold in < 5 seconds.
- [ ] Cold cache chart load (1 year of data) < 1 second on LAN.

Honest constraints (these are NOT bugs, document them in-app):

- Model IC ≈ 0.024; single-stock direction hit rate ≈ 51–52%. Signal is selection-grade, not certainty.
- Predictions for ETFs are constituent composites; coverage % is displayed and below 70% should be treated as "indicative only".
- Walk-forward retraining is v2; v1's weekly retrain is sufficient for monthly-regime markets.

## 11. Glossary

| Term | Meaning |
|---|---|
| **ticker / symbol** | qlib format: `SH600519`, `SZ000001` (exchange prefix + 6-digit code). Index: `SH000300`. |
| **score** | Model's predicted cross-sectional z-score of next-day relative return. Range typically ±0.05. Not a return percentage. |
| **signal date T** | Date the score was generated using features ≤ T close. Predicts return between T+1 close and T+2 close. |
| **target date T+2** | Date when the return predicted by `score[T]` is realized. Used in chart for visual alignment. |
| **IC** | Information Coefficient: Pearson correlation between score and realized return, daily cross-section, averaged. |
| **ICIR** | IC / std(IC). Risk-adjusted IC quality. |
| **recorder** | mlflow's run-level artifact container, holds pred.pkl, label.pkl, params.pkl, metrics. |
| **experiment** | mlflow's named grouping of recorders (e.g., `daily_cn_fresh`). |
| **mlruns/** | Directory mlflow uses to store experiments and recorders. |
| **qlib_data/** | Directory of binary-format OHLCV + calendars + instruments lists. |
| **dump_bin** | qlib utility that converts CSVs to binary feature storage. |
| **baostock** | Open-source CN data library; our chosen source over Yahoo (which is slow + has delisted-retry issues). |
| **Alpha158** | qlib's standard 158-factor handler (price/volume technical features). |
| **CSI300** | China Securities Index 300, the universe our model trains on. |
| **Tailscale** | WireGuard-based VPN we use for remote access without exposing public ports. |
| **MVP=D** | Decision label from Q2: full feature set in v1 (charts + signals + portfolio + ops UI). |

## 12. Open Questions

None blocking implementation. Items deferred to v1.x or v2 are noted inline.

---

**Next step**: After user review, invoke `superpowers:writing-plans` to produce a step-by-step implementation plan.
