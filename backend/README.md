# Qlib Companion · Backend

FastAPI + SQLAlchemy + qlib adapter. Single process; modular monolith.

## Setup

```bash
# uses existing qlib conda env
F:/Tools/Anaconda/envs/qlib/Scripts/pip.exe install -e '.[dev]'
```

## Run

```bash
F:/Tools/Anaconda/envs/qlib/python.exe -m uvicorn app.main:app --reload --port 8000
```

Open <http://localhost:8000/docs> for the API browser.

## Test

```bash
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest -v
```

## Migrations

```bash
F:/Tools/Anaconda/envs/qlib/python.exe -m alembic upgrade head
```

## Module layout

| Module | Purpose |
| --- | --- |
| `core/` | Config, DB, qlib adapter, logging, exceptions — shared kernel |
| `charts/` | OHLCV + prediction overlay (P1) |
| `ops/` | Health + (later) settings/jobs (P1: health only) |

P2/P3/P4 add `portfolio/`, `models/`, `data/`.
