# P1 · Foundation + Charts Vertical Slice — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the backend + frontend foundation, then ship one fully working end-to-end feature: open a browser, navigate to `/charts/SH600519`, see a real candlestick chart of Maotai with the model's prediction overlay. Boots in under 5 seconds, takes < 1s to load 1Y of data on LAN.

**Architecture:** Modular monolith. FastAPI (single process) serves both the JSON API and the built React SPA as static assets. SQLAlchemy 2.x async + Alembic ready for future tables. qlib is touched ONLY through `core/qlib_adapter.py`. React with Vite, TanStack Query for server state, Lightweight Charts for rendering. Both backend and frontend deps live in the existing `F:\Tools\Anaconda\envs\qlib` Python env and `frontend/node_modules`.

**Tech Stack:** Python 3.10 / FastAPI / SQLAlchemy 2.x async / aiosqlite / Alembic / Pydantic v2 / pydantic-settings / structlog / pytest + pytest-asyncio + httpx · Node 20 / Vite / React 18 / TypeScript 5 / TanStack Query 5 / Tailwind 3 / shadcn/ui / Lightweight Charts 4 / Vitest.

---

## File Layout Created by This Plan

```
backend/
├── pyproject.toml               # T0
├── alembic.ini                  # T5
├── alembic/
│   ├── env.py                   # T5
│   ├── script.py.mako           # T5 (default)
│   └── versions/0001_init.py    # T5
├── app/
│   ├── __init__.py
│   ├── main.py                  # T11
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py            # T1
│   │   ├── logging.py           # T2
│   │   ├── exceptions.py        # T3
│   │   ├── db.py                # T4
│   │   └── qlib_adapter.py      # T6
│   ├── charts/
│   │   ├── __init__.py
│   │   ├── schemas.py           # T7
│   │   ├── service.py           # T8
│   │   ├── router.py            # T9
│   │   └── tests/
│   │       ├── __init__.py
│   │       ├── test_service.py  # T8
│   │       └── test_router.py   # T9
│   └── ops/
│       ├── __init__.py
│       ├── router.py            # T10
│       └── schemas.py           # T10
└── tests/
    ├── __init__.py
    ├── conftest.py              # T4 (shared fixtures)
    └── integration/
        ├── __init__.py
        └── test_chart_e2e.py    # T19

frontend/
├── package.json                 # T12
├── vite.config.ts               # T12
├── tsconfig.json                # T12
├── tailwind.config.js           # T12
├── postcss.config.js            # T12
├── index.html                   # T12
├── src/
│   ├── main.tsx                 # T12
│   ├── App.tsx                  # T15
│   ├── api/
│   │   ├── client.ts            # T14
│   │   └── types.gen.ts         # T13 (generated)
│   ├── lib/
│   │   └── utils.ts             # T12
│   ├── components/
│   │   └── Layout.tsx           # T15
│   └── charts/
│       ├── ChartPage.tsx        # T17
│       ├── PredictionChart.tsx  # T16
│       └── hooks.ts             # T17
└── tests/
    └── PredictionChart.test.tsx # T16

.gitignore                       # T0 (update)
README.md                        # T20 (update)
```

---

## Task 0: Repo bootstrap + gitignore

**Files:**
- Create: `backend/`, `frontend/` (empty directories)
- Create: `backend/pyproject.toml`
- Modify: `.gitignore`

- [ ] **Step 1: Create directories**

```bash
cd /e/Projects/qlib/.claude/worktrees/eager-morse-4be0a4
mkdir -p backend/app/core backend/app/charts/tests backend/app/ops backend/tests/integration
mkdir -p frontend/src/api frontend/src/components frontend/src/charts frontend/src/lib frontend/tests
touch backend/app/__init__.py backend/app/core/__init__.py
touch backend/app/charts/__init__.py backend/app/charts/tests/__init__.py
touch backend/app/ops/__init__.py backend/tests/__init__.py backend/tests/integration/__init__.py
```

- [ ] **Step 2: Append entries to root .gitignore**

Append these lines to `.gitignore` at the worktree root:

```gitignore
# Backend
backend/.venv/
backend/__pycache__/
backend/**/__pycache__/
backend/*.egg-info/
backend/.pytest_cache/
backend/.coverage
backend/app.db
backend/app.db-journal

# Frontend
frontend/node_modules/
frontend/dist/
frontend/.vite/
frontend/coverage/

# Generated API types (kept in source for clarity, regenerated on schema change)
!frontend/src/api/types.gen.ts
```

- [ ] **Step 3: Create `backend/pyproject.toml`**

```toml
[project]
name = "qlib-companion-backend"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "sqlalchemy[asyncio]>=2.0",
    "aiosqlite>=0.20",
    "alembic>=1.13",
    "pydantic>=2.9",
    "pydantic-settings>=2.6",
    "structlog>=24.4",
    "python-multipart>=0.0.20",
    "httpx>=0.27",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.24",
    "pytest-cov>=6.0",
    "ruff>=0.7",
    "mypy>=1.13",
    "openapi-typescript>=7.4",
]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["app*"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["app", "tests"]
pythonpath = ["."]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B"]
ignore = ["E501"]
```

- [ ] **Step 4: Install deps into existing qlib env**

```bash
F:/Tools/Anaconda/envs/qlib/Scripts/pip.exe install -e 'backend[dev]'
```

Expected: succeeds; lists installed packages. If conflicts, the existing qlib env may have older pydantic — proceed; we pinned new floor versions.

- [ ] **Step 5: Smoke test — can we import the package?**

```bash
cd backend && F:/Tools/Anaconda/envs/qlib/Scripts/python.exe -c "import app; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 6: Commit**

```bash
git add backend/ .gitignore
git commit -m "feat(backend): scaffold pyproject + empty package skeleton"
```

---

## Task 1: core/config.py — Settings via pydantic-settings

**Files:**
- Create: `backend/app/core/config.py`
- Create: `backend/app/core/tests/__init__.py`
- Create: `backend/app/core/tests/test_config.py`

- [ ] **Step 1: Write the failing test**

`backend/app/core/tests/test_config.py`:
```python
from pathlib import Path
import os
from app.core.config import Settings


def test_settings_loads_defaults():
    s = Settings()
    assert s.api_port == 8000
    assert s.qlib_provider_uri.endswith("cn_data_bs")
    assert s.app_db_path.endswith("app.db")


def test_settings_overridable_by_env(monkeypatch):
    monkeypatch.setenv("QLIB_COMPANION_API_PORT", "9999")
    s = Settings()
    assert s.api_port == 9999


def test_resolved_paths_are_absolute():
    s = Settings()
    assert Path(s.qlib_provider_uri).expanduser().is_absolute()
```

`backend/app/core/__init__.py`: leave empty.

Create `backend/app/core/tests/__init__.py` empty.

- [ ] **Step 2: Run test, verify fail**

```bash
cd backend && F:/Tools/Anaconda/envs/qlib/Scripts/python.exe -m pytest app/core/tests/test_config.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.core.config'`.

- [ ] **Step 3: Implement**

`backend/app/core/config.py`:
```python
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="QLIB_COMPANION_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Storage paths
    qlib_provider_uri: str = "~/.qlib/qlib_data/cn_data_bs"
    qlib_region: str = "cn"
    mlruns_dir: str = "examples/mlruns"
    app_db_path: str = "backend/app.db"

    # Defaults
    default_experiment: str = "daily_cn_fresh"
    default_chart_window_days: int = 365

    @property
    def db_url(self) -> str:
        path = Path(self.app_db_path).expanduser().resolve()
        return f"sqlite+aiosqlite:///{path}"

    @property
    def qlib_data_dir(self) -> Path:
        return Path(self.qlib_provider_uri).expanduser().resolve()
```

- [ ] **Step 4: Verify tests pass**

```bash
cd backend && F:/Tools/Anaconda/envs/qlib/Scripts/python.exe -m pytest app/core/tests/test_config.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/core/config.py backend/app/core/tests/
git commit -m "feat(core): config via pydantic-settings"
```

---

## Task 2: core/logging.py — structlog setup

**Files:**
- Create: `backend/app/core/logging.py`
- Create: `backend/app/core/tests/test_logging.py`

- [ ] **Step 1: Write the failing test**

`backend/app/core/tests/test_logging.py`:
```python
import json
import logging
from app.core.logging import configure_logging, get_logger


def test_logger_emits_json(capsys):
    configure_logging(json_output=True)
    log = get_logger("test")
    log.info("hello", count=3)
    captured = capsys.readouterr().out
    data = json.loads(captured.strip().splitlines()[-1])
    assert data["event"] == "hello"
    assert data["count"] == 3
    assert data["level"] == "info"


def test_logger_emits_console(capsys):
    configure_logging(json_output=False)
    log = get_logger("test")
    log.warning("uh-oh", reason="x")
    out = capsys.readouterr().out
    assert "uh-oh" in out


def test_root_logger_respects_level():
    configure_logging(level="ERROR", json_output=True)
    assert logging.getLogger().level == logging.ERROR
```

- [ ] **Step 2: Verify fail**

```bash
cd backend && F:/Tools/Anaconda/envs/qlib/Scripts/python.exe -m pytest app/core/tests/test_logging.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement**

`backend/app/core/logging.py`:
```python
import logging
import sys
import structlog


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
        force=True,
    )

    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "app") -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
```

- [ ] **Step 4: Verify pass**

```bash
cd backend && F:/Tools/Anaconda/envs/qlib/Scripts/python.exe -m pytest app/core/tests/test_logging.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/core/logging.py backend/app/core/tests/test_logging.py
git commit -m "feat(core): structlog setup with JSON + console renderers"
```

---

## Task 3: core/exceptions.py — BusinessError hierarchy

**Files:**
- Create: `backend/app/core/exceptions.py`
- Create: `backend/app/core/tests/test_exceptions.py`

- [ ] **Step 1: Write the failing test**

`backend/app/core/tests/test_exceptions.py`:
```python
from app.core.exceptions import (
    BusinessError,
    NotFoundError,
    ConflictError,
    DependencyError,
)


def test_business_error_has_fields():
    e = BusinessError("bad input", code="bad_input", context={"field": "qty"})
    assert e.detail == "bad input"
    assert e.code == "bad_input"
    assert e.context == {"field": "qty"}
    assert e.http_status == 400


def test_not_found_is_404():
    e = NotFoundError("nope", code="missing")
    assert e.http_status == 404


def test_conflict_is_409():
    e = ConflictError("stale", code="stale_data")
    assert e.http_status == 409


def test_dependency_is_503():
    e = DependencyError("baostock down", code="upstream")
    assert e.http_status == 503


def test_dict_payload():
    e = NotFoundError("x", code="m", context={"id": 1})
    assert e.as_response_dict() == {"detail": "x", "code": "m", "context": {"id": 1}}
```

- [ ] **Step 2: Verify fail**

```bash
cd backend && F:/Tools/Anaconda/envs/qlib/Scripts/python.exe -m pytest app/core/tests/test_exceptions.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement**

`backend/app/core/exceptions.py`:
```python
from typing import Any


class BusinessError(Exception):
    """Base for all expected business-logic errors. Maps to HTTP 400 by default."""

    http_status: int = 400

    def __init__(self, detail: str, code: str, context: dict[str, Any] | None = None):
        super().__init__(detail)
        self.detail = detail
        self.code = code
        self.context = context or {}

    def as_response_dict(self) -> dict[str, Any]:
        return {"detail": self.detail, "code": self.code, "context": self.context}


class NotFoundError(BusinessError):
    http_status = 404


class ConflictError(BusinessError):
    http_status = 409


class DependencyError(BusinessError):
    """External dependency (qlib, baostock, file system) failed."""
    http_status = 503
```

- [ ] **Step 4: Verify pass**

```bash
cd backend && F:/Tools/Anaconda/envs/qlib/Scripts/python.exe -m pytest app/core/tests/test_exceptions.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/core/exceptions.py backend/app/core/tests/test_exceptions.py
git commit -m "feat(core): BusinessError hierarchy + HTTP status mapping"
```

---

## Task 4: core/db.py — async SQLAlchemy session

**Files:**
- Create: `backend/app/core/db.py`
- Create: `backend/tests/conftest.py`
- Create: `backend/app/core/tests/test_db.py`

- [ ] **Step 1: Write the failing test**

`backend/app/core/tests/test_db.py`:
```python
import pytest
from sqlalchemy import text
from app.core.db import create_engine_and_session, get_session


@pytest.mark.asyncio
async def test_session_factory_yields_working_session(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("QLIB_COMPANION_APP_DB_PATH", str(db_path))

    engine, session_maker = create_engine_and_session()
    async with session_maker() as session:
        result = await session.execute(text("SELECT 1"))
        assert result.scalar() == 1
    await engine.dispose()
```

`backend/tests/conftest.py`:
```python
import sys
from pathlib import Path
# Ensure `app` is importable from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

- [ ] **Step 2: Verify fail**

```bash
cd backend && F:/Tools/Anaconda/envs/qlib/Scripts/python.exe -m pytest app/core/tests/test_db.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement**

`backend/app/core/db.py`:
```python
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import Settings


class Base(DeclarativeBase):
    """Declarative base for all ORM models. Tables defined in module models.py files."""


def create_engine_and_session(
    settings: Settings | None = None,
) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    settings = settings or Settings()
    engine = create_async_engine(settings.db_url, echo=False, future=True)
    session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return engine, session_maker


# Singletons populated at startup in main.py lifespan
_engine: AsyncEngine | None = None
_session_maker: async_sessionmaker[AsyncSession] | None = None


def init_db_singletons(settings: Settings) -> None:
    global _engine, _session_maker
    _engine, _session_maker = create_engine_and_session(settings)


async def dispose_db_singletons() -> None:
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None


async def get_session() -> AsyncIterator[AsyncSession]:
    if _session_maker is None:
        raise RuntimeError("DB not initialized; call init_db_singletons() at startup")
    async with _session_maker() as session:
        yield session
```

- [ ] **Step 4: Verify pass**

```bash
cd backend && F:/Tools/Anaconda/envs/qlib/Scripts/python.exe -m pytest app/core/tests/test_db.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/core/db.py backend/app/core/tests/test_db.py backend/tests/conftest.py
git commit -m "feat(core): async SQLAlchemy session factory + singletons"
```

---

## Task 5: Alembic — initialize migrations with empty initial revision

**Files:**
- Create: `backend/alembic.ini`
- Create: `backend/alembic/env.py`
- Create: `backend/alembic/script.py.mako`
- Create: `backend/alembic/versions/0001_init.py`

- [ ] **Step 1: Initialize alembic (manual, since `alembic init` writes a default we then overwrite)**

```bash
cd backend && F:/Tools/Anaconda/envs/qlib/Scripts/python.exe -m alembic init alembic
```

This creates `alembic.ini` and `alembic/` directory. We will overwrite `env.py` next.

- [ ] **Step 2: Overwrite `backend/alembic.ini` with our settings**

```ini
[alembic]
script_location = alembic
prepend_sys_path = .
sqlalchemy.url = sqlite:///./app.db
file_template = %%(year)d%%(month).2d%%(day).2d_%%(hour).2d%%(minute).2d_%%(rev)s_%%(slug)s

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

- [ ] **Step 3: Overwrite `backend/alembic/env.py`**

```python
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool

from alembic import context
from app.core.config import Settings
from app.core.db import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

settings = Settings()
config.set_main_option("sqlalchemy.url", str(settings.db_url).replace("+aiosqlite", ""))

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

- [ ] **Step 4: Create initial empty migration**

`backend/alembic/versions/0001_init.py`:
```python
"""init

Revision ID: 0001
Revises:
Create Date: 2026-05-11

This migration intentionally creates no tables.
P1 has no SQLite-backed state; tables are added in P2 (portfolio) onward.
"""
from alembic import op  # noqa: F401
import sqlalchemy as sa  # noqa: F401


revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
```

- [ ] **Step 5: Verify migration runs against a fresh SQLite**

```bash
cd backend && rm -f app.db && F:/Tools/Anaconda/envs/qlib/Scripts/python.exe -m alembic upgrade head
```

Expected: output ends with `Running upgrade  -> 0001, init`. A `backend/app.db` file appears.

- [ ] **Step 6: Verify `alembic current` reports correct revision**

```bash
cd backend && F:/Tools/Anaconda/envs/qlib/Scripts/python.exe -m alembic current
```

Expected: `0001 (head)`.

- [ ] **Step 7: Commit**

```bash
git add backend/alembic.ini backend/alembic/
git commit -m "feat(db): alembic setup + empty initial migration"
```

---

## Task 6: core/qlib_adapter.py — the only qlib touchpoint

**Files:**
- Create: `backend/app/core/qlib_adapter.py`
- Create: `backend/app/core/tests/test_qlib_adapter.py`

- [ ] **Step 1: Write the failing test**

`backend/app/core/tests/test_qlib_adapter.py`:
```python
import pytest
import pandas as pd
from app.core.qlib_adapter import (
    init_qlib_once,
    get_ohlcv,
    get_calendar_end,
    get_csi300_instruments,
    load_pred,
    get_latest_recorder_id,
)


@pytest.fixture(scope="module", autouse=True)
def _init_qlib():
    init_qlib_once()


def test_get_csi300_returns_300ish():
    instruments = get_csi300_instruments()
    assert 200 <= len(instruments) <= 350
    assert all(s.startswith(("SH", "SZ")) for s in instruments)


def test_get_calendar_end_is_after_2025():
    end = get_calendar_end()
    assert end.year >= 2025


def test_get_ohlcv_returns_dataframe():
    end = get_calendar_end()
    df = get_ohlcv(["SH600519"], start="2025-01-01", end=str(end))
    assert isinstance(df, pd.DataFrame)
    assert {"$open", "$high", "$low", "$close", "$volume"}.issubset(df.columns)
    assert len(df) > 100


def test_load_pred_returns_series():
    rid = get_latest_recorder_id("daily_cn_fresh")
    pred = load_pred(rid)
    assert isinstance(pred, pd.Series)
    assert pred.index.nlevels == 2  # (datetime, instrument)
    assert len(pred) > 1000


def test_init_is_idempotent():
    init_qlib_once()
    init_qlib_once()  # second call should be a no-op, not raise
```

- [ ] **Step 2: Verify fail**

```bash
cd backend && F:/Tools/Anaconda/envs/qlib/Scripts/python.exe -m pytest app/core/tests/test_qlib_adapter.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement**

`backend/app/core/qlib_adapter.py`:
```python
from datetime import date
from pathlib import Path
from threading import Lock

import pandas as pd
import qlib
from qlib.constant import REG_CN, REG_US
from qlib.data import D
from qlib.workflow import R

from app.core.config import Settings
from app.core.exceptions import DependencyError, NotFoundError
from app.core.logging import get_logger

_log = get_logger("qlib_adapter")
_initialized = False
_lock = Lock()


def init_qlib_once(settings: Settings | None = None) -> None:
    """Idempotent qlib.init. Safe to call from many places."""
    global _initialized
    with _lock:
        if _initialized:
            return
        s = settings or Settings()
        region = REG_CN if s.qlib_region == "cn" else REG_US
        provider_uri = str(s.qlib_data_dir)
        if not Path(provider_uri).is_dir():
            raise DependencyError(
                f"qlib data not found at {provider_uri}",
                code="qlib_data_missing",
                context={"path": provider_uri},
            )
        qlib.init(provider_uri=provider_uri, region=region)
        _initialized = True
        _log.info("qlib_init_done", provider_uri=provider_uri, region=s.qlib_region)


def get_ohlcv(symbols: list[str], start: str, end: str, freq: str = "day") -> pd.DataFrame:
    """Return MultiIndex DataFrame (datetime x instrument) with columns $open/$high/$low/$close/$volume/$factor."""
    init_qlib_once()
    fields = ["$open", "$high", "$low", "$close", "$volume", "$factor"]
    df = D.features(instruments=symbols, fields=fields, start_time=start, end_time=end, freq=freq)
    if df is None or df.empty:
        raise NotFoundError(
            f"no ohlcv for {symbols} between {start} and {end}",
            code="ohlcv_empty",
            context={"symbols": symbols, "start": start, "end": end},
        )
    return df


def get_calendar_end() -> date:
    init_qlib_once()
    cal = D.calendar(freq="day")
    if not len(cal):
        raise DependencyError("empty trading calendar", code="calendar_empty")
    return pd.Timestamp(cal[-1]).date()


def get_csi300_instruments() -> list[str]:
    init_qlib_once()
    inst_dict = D.instruments("csi300")
    inst_list = D.list_instruments(instruments=inst_dict, as_list=True)
    return sorted(inst_list)


def get_latest_recorder_id(experiment_name: str) -> str:
    init_qlib_once()
    try:
        exp = R.get_exp(experiment_name=experiment_name)
    except Exception as e:
        raise NotFoundError(
            f"experiment '{experiment_name}' not found",
            code="experiment_missing",
            context={"name": experiment_name},
        ) from e
    recs = exp.list_recorders()
    if not recs:
        raise NotFoundError(
            f"no recorders in experiment '{experiment_name}'",
            code="no_recorders",
            context={"experiment": experiment_name},
        )
    for rid in sorted(recs, key=lambda k: recs[k].info["start_time"], reverse=True):
        try:
            r = exp.get_recorder(recorder_id=rid)
            r.load_object("pred.pkl")
            return rid
        except Exception:
            continue
    raise NotFoundError(
        f"no recorder with pred.pkl in '{experiment_name}'",
        code="no_pred_pkl",
        context={"experiment": experiment_name},
    )


def load_pred(recorder_id: str, experiment_name: str = "daily_cn_fresh") -> pd.Series:
    init_qlib_once()
    exp = R.get_exp(experiment_name=experiment_name)
    rec = exp.get_recorder(recorder_id=recorder_id)
    pred = rec.load_object("pred.pkl")
    if isinstance(pred, pd.DataFrame):
        pred = pred["score"]
    return pred
```

- [ ] **Step 4: Verify pass**

```bash
cd backend && F:/Tools/Anaconda/envs/qlib/Scripts/python.exe -m pytest app/core/tests/test_qlib_adapter.py -v
```

Expected: 5 passed. (Requires existing `~/.qlib/qlib_data/cn_data_bs` + `daily_cn_fresh` experiment in `examples/mlruns`. These exist from prior work.)

- [ ] **Step 5: Commit**

```bash
git add backend/app/core/qlib_adapter.py backend/app/core/tests/test_qlib_adapter.py
git commit -m "feat(core): qlib_adapter — the sole qlib touchpoint"
```

---

## Task 7: charts/schemas.py — Pydantic request/response types

**Files:**
- Create: `backend/app/charts/schemas.py`
- Create: `backend/app/charts/tests/test_schemas.py`

- [ ] **Step 1: Write the failing test**

`backend/app/charts/tests/test_schemas.py`:
```python
from datetime import date
from app.charts.schemas import CandleBar, PredictionBar, ChartPayload


def test_candle_bar_valid():
    bar = CandleBar(time="2026-05-08", open=10.0, high=11.0, low=9.5, close=10.5, volume=1000.0)
    assert bar.time == "2026-05-08"
    assert bar.close == 10.5


def test_prediction_bar_valid():
    pb = PredictionBar(time="2026-05-11", open=10.5, high=10.7, low=10.4, close=10.6, score=0.02)
    assert pb.score == 0.02


def test_chart_payload_assembles():
    payload = ChartPayload(
        symbol="SH600519",
        actual=[CandleBar(time="2026-05-08", open=1.0, high=2.0, low=0.5, close=1.5, volume=100.0)],
        predicted=[],
        forecast=[],
        meta={"last_actual_date": "2026-05-08", "experiment": "daily_cn_fresh"},
    )
    assert payload.symbol == "SH600519"
    assert len(payload.actual) == 1
```

- [ ] **Step 2: Verify fail**

```bash
cd backend && F:/Tools/Anaconda/envs/qlib/Scripts/python.exe -m pytest app/charts/tests/test_schemas.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement**

`backend/app/charts/schemas.py`:
```python
from typing import Any

from pydantic import BaseModel, Field


class CandleBar(BaseModel):
    time: str = Field(..., description="ISO date YYYY-MM-DD")
    open: float
    high: float
    low: float
    close: float
    volume: float


class PredictionBar(BaseModel):
    """Synthetic candle representing model's predicted close for a target date.

    open = prior actual close
    close = prior actual close * (1 + score_at_T-2)
    high/low computed with a small spread for visibility
    score field is the raw model score for cross-reference
    """

    time: str
    open: float
    high: float
    low: float
    close: float
    score: float


class ChartPayload(BaseModel):
    symbol: str
    actual: list[CandleBar]
    predicted: list[PredictionBar]
    forecast: list[PredictionBar] = Field(
        default_factory=list,
        description="Future-only predicted bars (dates beyond last actual)",
    )
    meta: dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 4: Verify pass**

```bash
cd backend && F:/Tools/Anaconda/envs/qlib/Scripts/python.exe -m pytest app/charts/tests/test_schemas.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/charts/schemas.py backend/app/charts/tests/test_schemas.py
git commit -m "feat(charts): Pydantic schemas for chart payload"
```

---

## Task 8: charts/service.py — assemble chart payload with proper time alignment

**Files:**
- Create: `backend/app/charts/service.py`
- Create: `backend/app/charts/tests/test_service.py`

- [ ] **Step 1: Write the failing test**

`backend/app/charts/tests/test_service.py`:
```python
import pytest
from app.core.qlib_adapter import init_qlib_once, get_calendar_end
from app.charts.service import get_chart


@pytest.fixture(scope="module", autouse=True)
def _init():
    init_qlib_once()


def test_get_chart_returns_actual_and_predicted():
    end = get_calendar_end()
    payload = get_chart(
        symbol="SH600519",
        start="2025-01-01",
        end=str(end),
        with_pred=True,
        experiment="daily_cn_fresh",
    )
    assert payload.symbol == "SH600519"
    assert len(payload.actual) > 100
    # predicted is shifted by 2 trading days, so should be (actual_len - 2)
    assert abs(len(payload.predicted) - (len(payload.actual) - 2)) <= 5
    # forecast contains future bars (1 or 2 of them)
    assert 0 <= len(payload.forecast) <= 2


def test_get_chart_without_pred_returns_only_actual():
    end = get_calendar_end()
    payload = get_chart(
        symbol="SH600519",
        start="2025-04-01",
        end=str(end),
        with_pred=False,
        experiment="daily_cn_fresh",
    )
    assert len(payload.actual) > 5
    assert len(payload.predicted) == 0
    assert len(payload.forecast) == 0


def test_get_chart_unknown_symbol_raises():
    from app.core.exceptions import NotFoundError

    with pytest.raises(NotFoundError):
        get_chart(symbol="SH999999", start="2025-01-01", end="2025-02-01", with_pred=False)


def test_predicted_bar_open_equals_prior_actual_close():
    end = get_calendar_end()
    payload = get_chart(
        symbol="SH600519",
        start="2025-04-01",
        end=str(end),
        with_pred=True,
        experiment="daily_cn_fresh",
    )
    actual_by_time = {b.time: b for b in payload.actual}
    actual_times = sorted(actual_by_time.keys())
    for pred_bar in payload.predicted[:5]:
        # find the actual bar 1 day before pred_bar.time in trading-day terms
        idx = actual_times.index(pred_bar.time)
        prior_actual = actual_by_time[actual_times[idx - 1]]
        assert abs(pred_bar.open - prior_actual.close) < 1e-6, (
            f"pred[{pred_bar.time}].open should equal actual[{actual_times[idx-1]}].close"
        )
```

- [ ] **Step 2: Verify fail**

```bash
cd backend && F:/Tools/Anaconda/envs/qlib/Scripts/python.exe -m pytest app/charts/tests/test_service.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement**

`backend/app/charts/service.py`:
```python
import pandas as pd

from app.charts.schemas import CandleBar, ChartPayload, PredictionBar
from app.core.config import Settings
from app.core.exceptions import NotFoundError
from app.core.qlib_adapter import (
    get_calendar_end,
    get_latest_recorder_id,
    get_ohlcv,
    init_qlib_once,
    load_pred,
)


def _next_trading_day_from_calendar(after: pd.Timestamp, calendar: list[pd.Timestamp]) -> pd.Timestamp | None:
    for d in calendar:
        if d > after:
            return d
    return None


def get_chart(
    symbol: str,
    start: str,
    end: str,
    with_pred: bool = True,
    experiment: str | None = None,
) -> ChartPayload:
    """Build a chart payload with actual + (optionally) predicted + forecast bars.

    Time alignment:
      - For each actual bar at trading day D[i] (i >= 2), the predicted bar at D[i]
        is computed from score[D[i-2]]: predicted_open = close[D[i-1]],
        predicted_close = close[D[i-1]] * (1 + score[D[i-2]]).
      - Forecast bars extend 1-2 trading days past last actual using the same logic.
    """
    init_qlib_once()
    settings = Settings()
    experiment = experiment or settings.default_experiment

    df = get_ohlcv([symbol], start=start, end=end)
    if df.empty:
        raise NotFoundError(
            f"no data for {symbol} between {start} and {end}",
            code="ohlcv_empty",
            context={"symbol": symbol},
        )

    # df has MultiIndex (datetime, instrument); pull the symbol slice
    if symbol not in df.index.get_level_values(1).unique():
        raise NotFoundError(
            f"symbol {symbol} not in dataset",
            code="symbol_missing",
            context={"symbol": symbol},
        )
    sub = df.xs(symbol, level=1).sort_index()
    actual: list[CandleBar] = [
        CandleBar(
            time=str(idx.date()),
            open=float(row["$open"]),
            high=float(row["$high"]),
            low=float(row["$low"]),
            close=float(row["$close"]),
            volume=float(row["$volume"]),
        )
        for idx, row in sub.iterrows()
    ]

    predicted: list[PredictionBar] = []
    forecast: list[PredictionBar] = []
    meta: dict = {
        "last_actual_date": actual[-1].time if actual else None,
        "with_pred": with_pred,
        "experiment": experiment,
    }

    if with_pred and len(actual) >= 3:
        recorder_id = get_latest_recorder_id(experiment)
        meta["recorder_id"] = recorder_id
        pred_series = load_pred(recorder_id, experiment_name=experiment)
        if symbol in pred_series.index.get_level_values(1).unique():
            scores = pred_series.xs(symbol, level=1).sort_index()
            score_map = {str(t.date()): float(v) for t, v in scores.items()}

            actual_dates = [b.time for b in actual]
            for i in range(2, len(actual)):
                sig_date = actual_dates[i - 2]
                if sig_date not in score_map:
                    continue
                sc = score_map[sig_date]
                prev_close = actual[i - 1].close
                pred_close = prev_close * (1 + sc)
                open_, close_ = prev_close, pred_close
                spread = max(0.001 * prev_close, abs(close_ - open_) * 0.08)
                predicted.append(
                    PredictionBar(
                        time=actual[i].time,
                        open=open_,
                        high=max(open_, close_) + spread,
                        low=min(open_, close_) - spread,
                        close=close_,
                        score=sc,
                    )
                )

            # Forecast: future bars beyond last actual.
            # score[last] predicts target = last + 2 trading days.
            # score[last-1] predicts target = last + 1 trading day.
            calendar_end = get_calendar_end()
            last_actual_date = pd.Timestamp(actual[-1].time)
            # naive future calendar using qlib's trading calendar for next 5 business days
            from qlib.data import D
            future_cal = [
                pd.Timestamp(d)
                for d in D.calendar(start_time=last_actual_date, end_time=last_actual_date + pd.Timedelta(days=10))
                if pd.Timestamp(d) > last_actual_date
            ][:2]
            if len(future_cal) >= 1 and len(actual) >= 2:
                sig_date_for_f1 = actual_dates[-2]
                if sig_date_for_f1 in score_map:
                    sc1 = score_map[sig_date_for_f1]
                    prev_close = actual[-1].close
                    close_f1 = prev_close * (1 + sc1)
                    spread = max(0.001 * prev_close, abs(close_f1 - prev_close) * 0.08)
                    forecast.append(
                        PredictionBar(
                            time=str(future_cal[0].date()),
                            open=prev_close,
                            high=max(prev_close, close_f1) + spread,
                            low=min(prev_close, close_f1) - spread,
                            close=close_f1,
                            score=sc1,
                        )
                    )
                    if len(future_cal) >= 2:
                        sig_date_for_f2 = actual_dates[-1]
                        if sig_date_for_f2 in score_map:
                            sc2 = score_map[sig_date_for_f2]
                            close_f2 = close_f1 * (1 + sc2)
                            spread2 = max(0.001 * close_f1, abs(close_f2 - close_f1) * 0.08)
                            forecast.append(
                                PredictionBar(
                                    time=str(future_cal[1].date()),
                                    open=close_f1,
                                    high=max(close_f1, close_f2) + spread2,
                                    low=min(close_f1, close_f2) - spread2,
                                    close=close_f2,
                                    score=sc2,
                                )
                            )
            meta["calendar_end"] = str(calendar_end)

    return ChartPayload(symbol=symbol, actual=actual, predicted=predicted, forecast=forecast, meta=meta)
```

- [ ] **Step 4: Verify pass**

```bash
cd backend && F:/Tools/Anaconda/envs/qlib/Scripts/python.exe -m pytest app/charts/tests/test_service.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/charts/service.py backend/app/charts/tests/test_service.py
git commit -m "feat(charts): get_chart service with T-2 prediction alignment + forecast bars"
```

---

## Task 9: charts/router.py — FastAPI endpoint + integration test

**Files:**
- Create: `backend/app/charts/router.py`
- Create: `backend/app/charts/tests/test_router.py`

- [ ] **Step 1: Write the failing test** (uses a local app builder so T9 can be verified independently of T11)

`backend/app/charts/tests/test_router.py`:
```python
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient

from app.charts.router import router as charts_router
from app.core.exceptions import BusinessError


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(charts_router, prefix="/api/charts", tags=["charts"])

    @app.exception_handler(BusinessError)
    async def biz_handler(_, exc: BusinessError):
        return JSONResponse(status_code=exc.http_status, content=exc.as_response_dict())

    return app


@pytest.fixture
async def client():
    app = _make_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_get_chart_200(client):
    r = await client.get(
        "/api/charts/SH600519",
        params={"start": "2025-04-01", "end": "2026-05-08", "with_pred": "true"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["symbol"] == "SH600519"
    assert isinstance(body["actual"], list)
    assert len(body["actual"]) > 5
    assert isinstance(body["predicted"], list)


@pytest.mark.asyncio
async def test_get_chart_unknown_symbol_404(client):
    r = await client.get(
        "/api/charts/SH999999",
        params={"start": "2025-04-01", "end": "2025-05-01", "with_pred": "false"},
    )
    assert r.status_code == 404
    body = r.json()
    assert body["code"] in {"ohlcv_empty", "symbol_missing"}
```

- [ ] **Step 2: Verify fail**

```bash
cd backend && F:/Tools/Anaconda/envs/qlib/Scripts/python.exe -m pytest app/charts/tests/test_router.py -v
```

Expected: `ImportError: app.charts.router` (we haven't written it yet).

- [ ] **Step 3: Implement**

`backend/app/charts/router.py`:
```python
from fastapi import APIRouter, Query

from app.charts.schemas import ChartPayload
from app.charts.service import get_chart

router = APIRouter()


@router.get("/{symbol}", response_model=ChartPayload)
async def chart(
    symbol: str,
    start: str = Query(..., description="ISO date YYYY-MM-DD"),
    end: str = Query(..., description="ISO date YYYY-MM-DD"),
    with_pred: bool = Query(default=True),
    experiment: str | None = Query(default=None),
) -> ChartPayload:
    return get_chart(symbol=symbol, start=start, end=end, with_pred=with_pred, experiment=experiment)
```

- [ ] **Step 4: Verify pass**

```bash
cd backend && F:/Tools/Anaconda/envs/qlib/Scripts/python.exe -m pytest app/charts/tests/test_router.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/charts/router.py backend/app/charts/tests/test_router.py
git commit -m "feat(charts): FastAPI router for /api/charts/{symbol}"
```

---

## Task 10: ops/router.py — /api/ops/health

**Files:**
- Create: `backend/app/ops/schemas.py`
- Create: `backend/app/ops/router.py`
- Create: `backend/app/ops/tests/__init__.py`
- Create: `backend/app/ops/tests/test_router.py`

- [ ] **Step 1: Write the failing test**

`backend/app/ops/tests/test_router.py`:
```python
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.ops.router import router as ops_router


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(ops_router, prefix="/api/ops", tags=["ops"])
    return app


@pytest.fixture
async def client():
    app = _make_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_health_ok(client):
    r = await client.get("/api/ops/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert "qlib_ready" in body
```

- [ ] **Step 2: Verify fail**

```bash
cd backend && F:/Tools/Anaconda/envs/qlib/Scripts/python.exe -m pytest app/ops/tests/test_router.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement**

`backend/app/ops/schemas.py`:
```python
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str
    qlib_ready: bool
    calendar_end: str | None = None
```

`backend/app/ops/router.py`:
```python
from fastapi import APIRouter

from app.core.qlib_adapter import get_calendar_end, init_qlib_once
from app.ops.schemas import HealthResponse

router = APIRouter()

APP_VERSION = "0.1.0"


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    qlib_ready = False
    cal_end = None
    try:
        init_qlib_once()
        cal_end = str(get_calendar_end())
        qlib_ready = True
    except Exception:
        pass
    return HealthResponse(
        status="ok",
        version=APP_VERSION,
        qlib_ready=qlib_ready,
        calendar_end=cal_end,
    )
```

`backend/app/ops/tests/__init__.py`: empty.

- [ ] **Step 4: Verify pass**

```bash
cd backend && F:/Tools/Anaconda/envs/qlib/Scripts/python.exe -m pytest app/ops/tests/test_router.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/ops/
git commit -m "feat(ops): /api/ops/health endpoint with qlib readiness"
```

---

## Task 11: app/main.py — wire everything together + lifespan

**Files:**
- Create: `backend/app/main.py`
- Create: `backend/tests/integration/test_app_boot.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/integration/test_app_boot.py`:
```python
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app


@pytest.fixture
async def client():
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_app_health(client):
    r = await client.get("/api/ops/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_openapi_docs_accessible(client):
    r = await client.get("/openapi.json")
    assert r.status_code == 200
    spec = r.json()
    paths = spec["paths"]
    assert "/api/charts/{symbol}" in paths
    assert "/api/ops/health" in paths


@pytest.mark.asyncio
async def test_business_error_returns_proper_status(client):
    r = await client.get(
        "/api/charts/SH999999",
        params={"start": "2025-01-01", "end": "2025-02-01"},
    )
    assert r.status_code == 404
    body = r.json()
    assert body["code"] in {"ohlcv_empty", "symbol_missing"}
    assert "detail" in body
```

- [ ] **Step 2: Verify fail**

```bash
cd backend && F:/Tools/Anaconda/envs/qlib/Scripts/python.exe -m pytest tests/integration/test_app_boot.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement**

`backend/app/main.py`:
```python
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.charts.router import router as charts_router
from app.core.config import Settings
from app.core.db import dispose_db_singletons, init_db_singletons
from app.core.exceptions import BusinessError
from app.core.logging import configure_logging, get_logger
from app.core.qlib_adapter import init_qlib_once
from app.ops.router import router as ops_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    configure_logging(json_output=True)
    log = get_logger("startup")
    init_db_singletons(settings)
    try:
        init_qlib_once(settings)
        log.info("qlib_ready")
    except Exception as e:
        log.warning("qlib_not_ready_at_boot", error=str(e))
    log.info("app_started", port=settings.api_port)
    yield
    await dispose_db_singletons()
    log.info("app_stopped")


def create_app() -> FastAPI:
    app = FastAPI(title="Qlib Companion", version="0.1.0", lifespan=lifespan)

    # CORS for local dev (Vite on :5173)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(BusinessError)
    async def business_error_handler(_, exc: BusinessError):
        return JSONResponse(status_code=exc.http_status, content=exc.as_response_dict())

    app.include_router(charts_router, prefix="/api/charts", tags=["charts"])
    app.include_router(ops_router, prefix="/api/ops", tags=["ops"])

    # Static serving of the built frontend (created in T18; tolerated if missing)
    static_dir = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="frontend")

    return app


app = create_app()
```

- [ ] **Step 4: Verify pass**

```bash
cd backend && F:/Tools/Anaconda/envs/qlib/Scripts/python.exe -m pytest tests/integration/test_app_boot.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Smoke-test the live server**

```bash
cd backend && F:/Tools/Anaconda/envs/qlib/Scripts/python.exe -m uvicorn app.main:app --port 8000 &
sleep 3
curl -s http://localhost:8000/api/ops/health | head -c 200
echo
# kill the server (windows: taskkill, but uvicorn ran in fg from another shell; for this test step, just verify the curl above worked)
```

Expected: JSON `{"status":"ok","version":"0.1.0","qlib_ready":true,"calendar_end":"2026-05-08"}` (date may differ).

- [ ] **Step 6: Commit**

```bash
git add backend/app/main.py backend/tests/integration/test_app_boot.py
git commit -m "feat(app): main.py wires routers, lifespan, error handler, static serve"
```

---

## Task 12: Frontend scaffold via Vite

**Files:**
- Create: `frontend/package.json`, `vite.config.ts`, `tsconfig.json`, `tailwind.config.js`, `postcss.config.js`, `index.html`, `src/main.tsx`, `src/lib/utils.ts`

- [ ] **Step 1: Create `frontend/package.json`**

```json
{
  "name": "qlib-companion-frontend",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "preview": "vite preview",
    "test": "vitest run",
    "typecheck": "tsc -b",
    "lint": "eslint src --max-warnings 0"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-router-dom": "^6.26.0",
    "@tanstack/react-query": "^5.59.0",
    "lightweight-charts": "^4.2.0",
    "clsx": "^2.1.1",
    "tailwind-merge": "^2.5.4"
  },
  "devDependencies": {
    "@types/react": "^18.3.12",
    "@types/react-dom": "^18.3.1",
    "@types/node": "^22.7.5",
    "@vitejs/plugin-react": "^4.3.3",
    "typescript": "^5.6.3",
    "vite": "^5.4.10",
    "vitest": "^2.1.4",
    "@testing-library/react": "^16.0.1",
    "@testing-library/jest-dom": "^6.6.2",
    "jsdom": "^25.0.1",
    "tailwindcss": "^3.4.14",
    "postcss": "^8.4.47",
    "autoprefixer": "^10.4.20"
  }
}
```

- [ ] **Step 2: Create `frontend/tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2023", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "moduleResolution": "Bundler",
    "skipLibCheck": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "esModuleInterop": true,
    "baseUrl": ".",
    "paths": { "@/*": ["src/*"] }
  },
  "include": ["src", "tests"]
}
```

- [ ] **Step 3: Create `frontend/vite.config.ts`**

```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { '@': path.resolve(__dirname, 'src') },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./tests/setup.ts'],
  },
});
```

- [ ] **Step 4: Create `frontend/tailwind.config.js`**

```javascript
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: 'class',
  theme: { extend: {} },
  plugins: [],
};
```

- [ ] **Step 5: Create `frontend/postcss.config.js`**

```javascript
export default {
  plugins: { tailwindcss: {}, autoprefixer: {} },
};
```

- [ ] **Step 6: Create `frontend/index.html`**

```html
<!doctype html>
<html lang="zh-CN" class="dark">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Qlib Companion</title>
  </head>
  <body class="bg-[#0d1117] text-[#e6edf3]">
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 7: Create `frontend/src/main.tsx`**

```tsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from '@/App';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';

const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 60_000, retry: 1 } },
});

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>,
);
```

- [ ] **Step 8: Create `frontend/src/index.css`**

```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

- [ ] **Step 9: Create `frontend/src/lib/utils.ts`**

```typescript
import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
```

- [ ] **Step 10: Create `frontend/tests/setup.ts`**

```typescript
import '@testing-library/jest-dom';
```

- [ ] **Step 11: Install + verify dev server boots**

```bash
cd frontend && npm install
npm run dev &
sleep 5
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:5173
# kill the dev server before continuing
```

Expected: `200` from curl. Then kill the dev process (Ctrl+C or `taskkill` the node).

- [ ] **Step 12: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/vite.config.ts frontend/tsconfig.json frontend/tailwind.config.js frontend/postcss.config.js frontend/index.html frontend/src/main.tsx frontend/src/index.css frontend/src/lib/utils.ts frontend/tests/setup.ts
git commit -m "feat(frontend): Vite + React + Tailwind + TanStack Query scaffold"
```

---

## Task 13: openapi-typescript codegen — auto-generated API types

**Files:**
- Modify: `frontend/package.json` (add `gen:api` script)
- Create: `frontend/src/api/types.gen.ts` (generated, committed)

- [ ] **Step 1: Add `gen:api` script to `frontend/package.json`**

In the `scripts` block, add (after `"test"`):

```json
"gen:api": "openapi-typescript http://localhost:8000/openapi.json -o src/api/types.gen.ts"
```

Then install the tool as a frontend dev dep too (it was also added to backend dev deps in T0; here we install the JS one):

```bash
cd frontend && npm install --save-dev openapi-typescript
```

- [ ] **Step 2: Start the backend, run codegen, stop the backend**

```bash
# Terminal 1
cd backend && F:/Tools/Anaconda/envs/qlib/Scripts/python.exe -m uvicorn app.main:app --port 8000 &
sleep 4
# Terminal 2 (same shell, since we backgrounded)
cd frontend && npm run gen:api
# stop the backend
kill %1   # or use taskkill on Windows; uvicorn process should release port 8000
```

Expected: `frontend/src/api/types.gen.ts` is created (file size > 1 KB, contains `paths` / `components` / `schemas` namespaces).

- [ ] **Step 3: Sanity-check generated file structure**

```bash
head -30 frontend/src/api/types.gen.ts
grep -c "operations\|components\|paths" frontend/src/api/types.gen.ts
```

Expected: file starts with `/**` and `export interface paths`. grep count >= 3.

- [ ] **Step 4: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/src/api/types.gen.ts
git commit -m "feat(frontend): openapi-typescript codegen for /api types"
```

---

## Task 14: Typed API client

**Files:**
- Create: `frontend/src/api/client.ts`

- [ ] **Step 1: Implement (no test — wrapper over fetch is too thin to TDD; covered by T17 hook tests)**

`frontend/src/api/client.ts`:
```typescript
import type { paths } from '@/api/types.gen';

const BASE = ''; // empty in production (same origin); Vite proxy handles /api in dev

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    public detail: string,
    public context: Record<string, unknown> = {},
  ) {
    super(detail);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    ...init,
  });
  if (!res.ok) {
    let body: { detail?: string; code?: string; context?: Record<string, unknown> } = {};
    try {
      body = await res.json();
    } catch {
      /* ignore */
    }
    throw new ApiError(res.status, body.code ?? 'unknown', body.detail ?? res.statusText, body.context ?? {});
  }
  return res.json() as Promise<T>;
}

export const api = {
  charts: {
    get: (
      symbol: string,
      params: { start: string; end: string; with_pred?: boolean; experiment?: string },
    ) => {
      const q = new URLSearchParams({
        start: params.start,
        end: params.end,
        with_pred: String(params.with_pred ?? true),
        ...(params.experiment ? { experiment: params.experiment } : {}),
      });
      type R = paths['/api/charts/{symbol}']['get']['responses']['200']['content']['application/json'];
      return request<R>(`/api/charts/${encodeURIComponent(symbol)}?${q.toString()}`);
    },
  },
  ops: {
    health: () => {
      type R = paths['/api/ops/health']['get']['responses']['200']['content']['application/json'];
      return request<R>('/api/ops/health');
    },
  },
};
```

- [ ] **Step 2: typecheck**

```bash
cd frontend && npm run typecheck
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/api/client.ts
git commit -m "feat(frontend): typed api client over openapi-generated types"
```

---

## Task 15: App shell + routing + layout

**Files:**
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/components/Layout.tsx`

- [ ] **Step 1: Implement `frontend/src/components/Layout.tsx`**

```tsx
import { Link, Outlet } from 'react-router-dom';
import { cn } from '@/lib/utils';

export default function Layout() {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-[#30363d] px-6 py-3 flex items-center gap-6">
        <span className="font-semibold">Qlib Companion</span>
        <nav className="flex gap-4 text-sm">
          <NavLink to="/">Dashboard</NavLink>
          <NavLink to="/charts/SH600519">Charts</NavLink>
        </nav>
      </header>
      <main className="flex-1 p-6">
        <Outlet />
      </main>
    </div>
  );
}

function NavLink({ to, children }: { to: string; children: React.ReactNode }) {
  return (
    <Link to={to} className={cn('text-[#8b949e] hover:text-[#e6edf3] transition')}>
      {children}
    </Link>
  );
}
```

- [ ] **Step 2: Implement `frontend/src/App.tsx`**

```tsx
import { Route, Routes } from 'react-router-dom';
import Layout from '@/components/Layout';
import ChartPage from '@/charts/ChartPage';

function Home() {
  return (
    <div className="space-y-2">
      <h1 className="text-xl font-semibold">Dashboard</h1>
      <p className="text-[#8b949e] text-sm">
        Open a chart by URL: <code>/charts/SH600519</code> · <code>/charts/SZ000001</code>
      </p>
    </div>
  );
}

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Home />} />
        <Route path="/charts/:symbol" element={<ChartPage />} />
      </Route>
    </Routes>
  );
}
```

(ChartPage is a placeholder import for now; Task 17 implements it. To keep TS happy until then, create a stub:)

`frontend/src/charts/ChartPage.tsx` (temporary stub):
```tsx
export default function ChartPage() {
  return <div>chart page coming…</div>;
}
```

- [ ] **Step 3: Verify typecheck + dev server**

```bash
cd frontend && npm run typecheck && npm run dev &
sleep 4
curl -s http://localhost:5173 | head -c 200
# kill dev server
```

Expected: typecheck passes; curl returns the index HTML mentioning `<div id="root"></div>`.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/App.tsx frontend/src/components/ frontend/src/charts/ChartPage.tsx
git commit -m "feat(frontend): app shell, layout, router routes"
```

---

## Task 16: PredictionChart component (the killer chart, properly built)

**Files:**
- Create: `frontend/src/charts/PredictionChart.tsx`
- Create: `frontend/tests/PredictionChart.test.tsx`

> **Note:** The brainstorming-era `chart-overlay-v2.html` is NOT reused. This implementation is fresh: cleaner state management, no `setData` rebuild on opacity slide, proper React refs, separator line dropped (uses `addMarker` instead), and component is unit-tested.

- [ ] **Step 1: Write the failing test**

`frontend/tests/PredictionChart.test.tsx`:
```tsx
import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import PredictionChart from '@/charts/PredictionChart';

// Lightweight Charts is canvas-based and doesn't render text into the DOM.
// We test that the component mounts, accepts data, and renders the legend / toolbar.

vi.mock('lightweight-charts', () => ({
  createChart: () => ({
    addCandlestickSeries: () => ({
      setData: vi.fn(),
      applyOptions: vi.fn(),
      setMarkers: vi.fn(),
    }),
    timeScale: () => ({ fitContent: vi.fn(), setVisibleRange: vi.fn() }),
    remove: vi.fn(),
  }),
}));

const fakeActual = Array.from({ length: 10 }, (_, i) => ({
  time: `2026-04-${(20 + i).toString().padStart(2, '0')}`,
  open: 100 + i,
  high: 102 + i,
  low: 99 + i,
  close: 101 + i,
  volume: 1000,
}));

const fakePred = fakeActual.slice(2).map((b, i) => ({
  time: b.time,
  open: fakeActual[i + 1].close,
  close: fakeActual[i + 1].close * (1 + 0.005),
  high: 0,
  low: 0,
  score: 0.005,
}));

describe('PredictionChart', () => {
  it('renders toggles and opacity slider', () => {
    render(
      <PredictionChart
        symbol="SH600519"
        actual={fakeActual}
        predicted={fakePred}
        forecast={[]}
        lastActualDate="2026-04-29"
      />,
    );
    expect(screen.getByLabelText(/实际/)).toBeInTheDocument();
    expect(screen.getByLabelText(/预测/)).toBeInTheDocument();
    expect(screen.getByLabelText(/透明度/)).toBeInTheDocument();
  });

  it('renders symbol heading', () => {
    render(
      <PredictionChart
        symbol="SH600519"
        actual={fakeActual}
        predicted={[]}
        forecast={[]}
        lastActualDate="2026-04-29"
      />,
    );
    expect(screen.getByText(/SH600519/)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Verify fail**

```bash
cd frontend && npm run test
```

Expected: ImportError or "cannot find module" for PredictionChart.

- [ ] **Step 3: Implement**

`frontend/src/charts/PredictionChart.tsx`:
```tsx
import { useEffect, useMemo, useRef, useState } from 'react';
import { createChart, type IChartApi, type ISeriesApi } from 'lightweight-charts';
import { cn } from '@/lib/utils';

export interface CandleBar {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface PredictionBar {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  score: number;
}

interface Props {
  symbol: string;
  actual: CandleBar[];
  predicted: PredictionBar[];
  forecast: PredictionBar[];
  lastActualDate: string;
}

const ACTUAL_UP = '#26a69a';
const ACTUAL_DN = '#ef5350';
const PRED_BULL = (a: number) => `rgba(59,130,246,${a})`;
const PRED_BEAR = (a: number) => `rgba(250,204,21,${a})`;

export default function PredictionChart({ symbol, actual, predicted, forecast, lastActualDate }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const actualSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const predSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);

  const [showActual, setShowActual] = useState(true);
  const [showPred, setShowPred] = useState(true);
  const [opacity, setOpacity] = useState(40);

  // Build predicted bar dataset incl. forecast, with per-bar coloring derived from `score`.
  const styledPredBars = useMemo(() => {
    const a = opacity / 100;
    const border = Math.min(1, a + 0.3);
    return [...predicted, ...forecast].map(b => {
      const bull = b.score > 0;
      return {
        ...b,
        color: bull ? PRED_BULL(a) : PRED_BEAR(a),
        borderColor: bull ? PRED_BULL(border) : PRED_BEAR(border),
        wickColor: bull ? PRED_BULL(border) : PRED_BEAR(border),
      };
    });
  }, [predicted, forecast, opacity]);

  // Mount chart once
  useEffect(() => {
    if (!containerRef.current) return;
    const chart = createChart(containerRef.current, {
      layout: { background: { color: '#0d1117' }, textColor: '#e6edf3' } as never,
      grid: { vertLines: { color: '#21262d' }, horzLines: { color: '#21262d' } },
      rightPriceScale: { borderColor: '#30363d' },
      timeScale: { borderColor: '#30363d', rightOffset: 8 },
      crosshair: { mode: 1 },
      autoSize: true,
    });
    chartRef.current = chart;
    actualSeriesRef.current = chart.addCandlestickSeries({
      upColor: ACTUAL_UP,
      downColor: ACTUAL_DN,
      borderUpColor: ACTUAL_UP,
      borderDownColor: ACTUAL_DN,
      wickUpColor: ACTUAL_UP,
      wickDownColor: ACTUAL_DN,
    });
    predSeriesRef.current = chart.addCandlestickSeries({});
    return () => {
      chart.remove();
      chartRef.current = null;
      actualSeriesRef.current = null;
      predSeriesRef.current = null;
    };
  }, []);

  // Sync actual data
  useEffect(() => {
    actualSeriesRef.current?.setData(actual);
    chartRef.current?.timeScale().fitContent();
    // Marker on last actual day pointing right (future starts here)
    if (actual.length) {
      actualSeriesRef.current?.setMarkers?.([
        {
          time: lastActualDate,
          position: 'aboveBar',
          color: '#ff9800',
          shape: 'arrowRight',
          text: '→ 未来',
        },
      ]);
    }
  }, [actual, lastActualDate]);

  // Sync predicted data + opacity
  useEffect(() => {
    predSeriesRef.current?.setData(styledPredBars);
  }, [styledPredBars]);

  // Toggle visibility
  useEffect(() => {
    actualSeriesRef.current?.applyOptions({ visible: showActual });
  }, [showActual]);
  useEffect(() => {
    predSeriesRef.current?.applyOptions({ visible: showPred });
  }, [showPred]);

  return (
    <div className="space-y-3">
      <h2 className="text-lg font-semibold">{symbol}</h2>
      <div className="flex flex-wrap items-center gap-4 text-sm">
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={showActual}
            onChange={e => setShowActual(e.target.checked)}
            aria-label="实际 K 线"
          />
          实际 K 线
        </label>
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={showPred}
            onChange={e => setShowPred(e.target.checked)}
            aria-label="预测 K 线"
          />
          预测 K 线
        </label>
        <label className="flex items-center gap-2">
          <span>预测透明度</span>
          <input
            type="range"
            min={0}
            max={100}
            value={opacity}
            onChange={e => setOpacity(Number(e.target.value))}
            aria-label="预测透明度"
            className="w-40"
          />
          <span className="w-10 text-right">{opacity}%</span>
        </label>
      </div>
      <div
        ref={containerRef}
        className={cn('w-full h-[480px] border border-[#30363d] rounded-lg overflow-hidden')}
      />
    </div>
  );
}
```

- [ ] **Step 4: Verify pass**

```bash
cd frontend && npm run test && npm run typecheck
```

Expected: tests pass; typecheck clean.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/charts/PredictionChart.tsx frontend/tests/PredictionChart.test.tsx
git commit -m "feat(frontend): PredictionChart component with toggles + opacity"
```

---

## Task 17: ChartPage + useChart hook — fetch + render

**Files:**
- Create: `frontend/src/charts/hooks.ts`
- Modify: `frontend/src/charts/ChartPage.tsx` (replace stub)

- [ ] **Step 1: Implement `frontend/src/charts/hooks.ts`**

```typescript
import { useQuery } from '@tanstack/react-query';
import { api, ApiError } from '@/api/client';

interface ChartArgs {
  symbol: string;
  start: string;
  end: string;
  withPred?: boolean;
}

export function useChart({ symbol, start, end, withPred = true }: ChartArgs) {
  return useQuery({
    queryKey: ['chart', symbol, start, end, withPred],
    queryFn: () => api.charts.get(symbol, { start, end, with_pred: withPred }),
    enabled: !!symbol,
    staleTime: 5 * 60_000,
    retry: (count, err) => (err instanceof ApiError && err.status === 404 ? false : count < 2),
  });
}
```

- [ ] **Step 2: Implement `frontend/src/charts/ChartPage.tsx`** (overwrite the stub)

```tsx
import { useParams } from 'react-router-dom';
import { useChart } from '@/charts/hooks';
import PredictionChart from '@/charts/PredictionChart';

function defaultDateRange(): { start: string; end: string } {
  const end = new Date();
  const start = new Date();
  start.setFullYear(end.getFullYear() - 1);
  return { start: start.toISOString().slice(0, 10), end: end.toISOString().slice(0, 10) };
}

export default function ChartPage() {
  const { symbol = '' } = useParams<{ symbol: string }>();
  const { start, end } = defaultDateRange();
  const { data, isPending, error } = useChart({ symbol, start, end, withPred: true });

  if (isPending) {
    return <div className="text-[#8b949e]">Loading {symbol}…</div>;
  }
  if (error) {
    return (
      <div className="text-red-400">
        Failed to load {symbol}: {(error as Error).message}
      </div>
    );
  }
  if (!data) return null;

  return (
    <PredictionChart
      symbol={data.symbol}
      actual={data.actual}
      predicted={data.predicted}
      forecast={data.forecast}
      lastActualDate={data.meta.last_actual_date ?? ''}
    />
  );
}
```

- [ ] **Step 3: typecheck**

```bash
cd frontend && npm run typecheck
```

Expected: clean.

- [ ] **Step 4: Manual end-to-end check**

```bash
# Terminal 1
cd backend && F:/Tools/Anaconda/envs/qlib/Scripts/python.exe -m uvicorn app.main:app --port 8000 &
# Terminal 2
cd frontend && npm run dev &
sleep 5
```

In the browser open `http://localhost:5173/charts/SH600519`. Verify:
- Chart renders within 2 seconds
- Toggling "实际 K 线" / "预测 K 线" hides/shows respective candle series
- Sliding opacity recolors predicted bars without flicker
- Future marker (`→ 未来`) sits on the last actual candle

Stop both servers when done.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/charts/hooks.ts frontend/src/charts/ChartPage.tsx
git commit -m "feat(frontend): ChartPage wires useChart hook into PredictionChart"
```

---

## Task 18: Backend serves built frontend (single-binary deploy)

**Files:**
- No new code — verifies that the static mount in `app/main.py` works against `frontend/dist/`.

- [ ] **Step 1: Build the frontend**

```bash
cd frontend && npm run build
```

Expected: `frontend/dist/` contains `index.html` + `assets/` directory.

- [ ] **Step 2: Start backend in production mode (no dev proxy)**

```bash
cd backend && F:/Tools/Anaconda/envs/qlib/Scripts/python.exe -m uvicorn app.main:app --port 8000 &
sleep 3
```

- [ ] **Step 3: Verify backend serves the SPA at `/`**

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/
curl -s http://localhost:8000/ | grep -c "<div id=\"root\""
```

Expected: status `200`; grep count `1`.

- [ ] **Step 4: Verify deep-link works (SPA fallback)**

`StaticFiles(html=True)` already serves `index.html` for missing files. Verify:

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/charts/SH600519
```

Expected: `200` (returns index.html since the path has no static file; React Router then handles routing client-side).

- [ ] **Step 5: Smoke-check chart API still works from same origin**

```bash
curl -s "http://localhost:8000/api/charts/SH600519?start=2026-04-01&end=2026-05-08&with_pred=true" | head -c 200
```

Expected: JSON starting with `{"symbol":"SH600519"`.

- [ ] **Step 6: Stop backend**

```bash
# windows: find pid and taskkill, or close the terminal
```

- [ ] **Step 7: Commit (only build artifacts; no code change but pin npm-lock)**

```bash
# nothing to commit unless lockfiles changed; verify with git status
git status
```

If git is clean, skip commit. If `package-lock.json` updated, commit it:
```bash
git add frontend/package-lock.json
git commit -m "chore(frontend): lockfile from prod build"
```

---

## Task 19: End-to-end smoke test (pytest)

**Files:**
- Create: `backend/tests/integration/test_chart_e2e.py` (new file; T11's file was `test_app_boot.py`)

- [ ] **Step 1: Write the e2e tests**

`backend/tests/integration/test_chart_e2e.py`:
```python
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app


@pytest.fixture
async def client():
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_chart_real_data_for_maotai(client):
    r = await client.get(
        "/api/charts/SH600519",
        params={"start": "2025-01-01", "end": "2026-05-08", "with_pred": "true"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["symbol"] == "SH600519"
    actual = body["actual"]
    predicted = body["predicted"]
    assert len(actual) > 200
    assert len(predicted) > 0
    # last actual should match start of forecast window
    last_actual = actual[-1]["time"]
    assert body["meta"]["last_actual_date"] == last_actual


@pytest.mark.asyncio
async def test_chart_alignment_invariant(client):
    """Property: every predicted bar's open == prior actual bar's close (within float epsilon)."""
    r = await client.get(
        "/api/charts/SH600519",
        params={"start": "2025-04-01", "end": "2026-05-08", "with_pred": "true"},
    )
    body = r.json()
    actual_by_time = {b["time"]: b for b in body["actual"]}
    actual_times = sorted(actual_by_time.keys())
    for pb in body["predicted"]:
        idx = actual_times.index(pb["time"])
        prior = actual_by_time[actual_times[idx - 1]]
        assert abs(pb["open"] - prior["close"]) < 1e-6
```

- [ ] **Step 2: Run**

```bash
cd backend && F:/Tools/Anaconda/envs/qlib/Scripts/python.exe -m pytest tests/integration/test_chart_e2e.py -v
```

Expected: 2 passed (plus the 3 from T11). Total in this file: 5 passed.

- [ ] **Step 3: Run the entire test suite to catch regressions**

```bash
cd backend && F:/Tools/Anaconda/envs/qlib/Scripts/python.exe -m pytest -v
```

Expected: all green. Total ~20-25 tests.

- [ ] **Step 4: Commit**

```bash
git add backend/tests/integration/test_chart_e2e.py
git commit -m "test(charts): e2e — real data shape + T-2 alignment invariant"
```

---

## Task 20: README + dev quickstart docs

**Files:**
- Create: `backend/README.md`
- Create: `frontend/README.md`
- Modify: root `README.md` (append project section)

- [ ] **Step 1: Create `backend/README.md`**

```markdown
# Qlib Companion · Backend

FastAPI + SQLAlchemy + qlib adapter. Single process; modular monolith.

## Setup

```bash
# uses existing qlib conda env
F:/Tools/Anaconda/envs/qlib/Scripts/pip.exe install -e '.[dev]'
```

## Run

```bash
F:/Tools/Anaconda/envs/qlib/Scripts/python.exe -m uvicorn app.main:app --reload --port 8000
```

Open <http://localhost:8000/docs> for the API browser.

## Test

```bash
F:/Tools/Anaconda/envs/qlib/Scripts/python.exe -m pytest -v
```

## Migrations

```bash
F:/Tools/Anaconda/envs/qlib/Scripts/python.exe -m alembic upgrade head
```

## Module layout

| Module | Purpose |
| --- | --- |
| `core/` | Config, DB, qlib adapter, logging, exceptions — shared kernel |
| `charts/` | OHLCV + prediction overlay (P1) |
| `ops/` | Health + (later) settings/jobs (P1: health only) |

P2/P3/P4 add `portfolio/`, `models/`, `data/`.
```

- [ ] **Step 2: Create `frontend/README.md`**

```markdown
# Qlib Companion · Frontend

React 18 + Vite + Tailwind + shadcn-style components + TanStack Query + Lightweight Charts.

## Setup

```bash
npm install
```

## Dev

```bash
npm run dev          # http://localhost:5173 (proxies /api to :8000)
npm run gen:api      # regenerate types from backend /openapi.json
npm test
npm run typecheck
```

## Build

```bash
npm run build        # outputs frontend/dist/, served by backend
```

## Routes

| Route | Page |
| --- | --- |
| `/` | Dashboard (P1: stub) |
| `/charts/:symbol` | Single-ticker chart with prediction overlay |

P2-P4 add `/picks`, `/portfolio`, `/ops`.
```

- [ ] **Step 3: Append section to root `README.md`** (the existing README is qlib's; we add an "App" section near the top)

Insert after the existing top-level heading:

```markdown
## Qlib Companion App

This worktree contains a web app on top of the qlib pipeline. See [docs/superpowers/specs/2026-05-11-qlib-trading-companion-design.md](docs/superpowers/specs/2026-05-11-qlib-trading-companion-design.md) for design and [docs/superpowers/plans/2026-05-11-p1-foundation-charts-slice.md](docs/superpowers/plans/2026-05-11-p1-foundation-charts-slice.md) for implementation plan P1.

Quick start (after running `update_qlib_data.ps1` once to seed data):

```bash
# Backend
cd backend && F:/Tools/Anaconda/envs/qlib/Scripts/python.exe -m uvicorn app.main:app --port 8000

# Frontend (in another terminal)
cd frontend && npm install && npm run dev
```

Open <http://localhost:5173/charts/SH600519> for a sample chart.
```

(Skip this step if the existing README doesn't accept Markdown additions cleanly — instead just create the two READMEs in backend/ and frontend/.)

- [ ] **Step 4: Commit**

```bash
git add backend/README.md frontend/README.md README.md
git commit -m "docs: P1 quickstart for backend + frontend"
```

---

## Task 21: P1 acceptance verification

This is the final "v1 of P1 done" gate. No new code; runs everything to confirm.

- [ ] **Step 1: Clean rebuild from scratch**

```bash
# Backend
cd backend && rm -f app.db && F:/Tools/Anaconda/envs/qlib/Scripts/python.exe -m alembic upgrade head

# Frontend
cd ../frontend && rm -rf dist node_modules && npm install && npm run build
```

Expected: both succeed without warnings.

- [ ] **Step 2: Run all backend tests**

```bash
cd ../backend && F:/Tools/Anaconda/envs/qlib/Scripts/python.exe -m pytest -v --tb=short
```

Expected: all green; ~25 tests.

- [ ] **Step 3: Run all frontend tests + typecheck**

```bash
cd ../frontend && npm run test && npm run typecheck
```

Expected: all green.

- [ ] **Step 4: Manual smoke (single integrated server)**

```bash
cd ../backend && F:/Tools/Anaconda/envs/qlib/Scripts/python.exe -m uvicorn app.main:app --port 8000 &
sleep 4
curl -s http://localhost:8000/api/ops/health
# browser: open http://localhost:8000/charts/SH600519
# verify chart renders within 2s
```

- [ ] **Step 5: Verify acceptance criteria from spec section 10 that apply to P1**

| Criterion | P1 covers |
| --- | --- |
| Backend boots from cold in < 5 seconds | ✅ verify in step 4 |
| Cold cache chart load (1 year) < 1 second on LAN | ✅ verify in browser DevTools Network panel |
| OpenAPI spec at `/docs` is current | ✅ visit `http://localhost:8000/docs` |
| No `print()` or `console.log()` in non-test code | ✅ run `grep -r "print(" backend/app | grep -v test` (should be empty) and same for console.log |
| Test coverage ≥ 70% on `service.py` files | ✅ `pytest --cov=app.charts --cov=app.core` |

- [ ] **Step 6: Final commit (only if any of step-5 fixes needed)**

```bash
git status
# if clean: nothing to do
# if dirty: commit fixes with message "chore: P1 acceptance fixes"
```

---

## Summary

After T0–T21 you have:

- A clean backend package importable from the existing qlib conda env, with structlog, pydantic-settings, async SQLAlchemy session factory, Alembic ready for future migrations.
- A `qlib_adapter` that is the **only** place qlib is imported. All other code calls its 5 public functions.
- A `charts/` module: schemas → service (handles T-2 alignment + forward forecast bars) → router. 4 service tests + 2 router tests + 2 e2e tests.
- An `ops/health` endpoint that the future Tailscale health check can use.
- FastAPI lifespan that boots cleanly and degrades gracefully when qlib data is missing.
- Vite + React + TanStack Query + Tailwind frontend scaffold with a single working page (`/charts/:symbol`) rendering the dual-layer candle + prediction overlay + forward bars + opacity slider.
- Typed end-to-end via openapi-typescript generated types.
- README + dev quickstart for both halves.

P2 (portfolio module + pages) can start immediately after P1 merges. It depends only on the `core/` foundation already shipped.
