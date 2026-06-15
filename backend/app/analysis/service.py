"""In-memory job tracking + background worker for AI analysis.
Mirrors app/inference/service.py. Single-process backend assumption."""
from __future__ import annotations

import logging
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from app.analysis import store
from app.analysis.llm import analyze_one, make_client, _resolve_key
from app.analysis.schemas import AnalysisJob, AnalysisStatus, TriggerResponse
from app.analysis.sources import fetch_news, fetch_notices
from app.core.config import Settings

log = logging.getLogger(__name__)

_MAX_JOBS = 50
_JOBS: "OrderedDict[str, AnalysisJob]" = OrderedDict()
_ACTIVE_JOB_ID: str | None = None
_LOCK = threading.Lock()
_LAST_RUN_AT: str | None = None
_LAST_SUCCESS_AT: str | None = None
_LAST_ERROR: str | None = None

_CONCURRENCY = 4


# --- config resolvers (monkeypatchable in tests) --------------------------
def _is_enabled(s: Settings) -> bool:
    # Enabled when the toggle is on AND the *active provider's* key is resolvable.
    return bool(s.ai_analysis_enabled and _resolve_key(s))

def _db_path(s: Settings) -> str:
    return str(Path(s.app_db_path).expanduser().resolve())

def _top_n(s: Settings) -> int:
    return s.ai_analysis_top_n


def _remember(job_id: str, job: AnalysisJob) -> None:
    _JOBS[job_id] = job
    _JOBS.move_to_end(job_id)
    while len(_JOBS) > _MAX_JOBS:
        _JOBS.popitem(last=False)


def get_active_job() -> AnalysisJob | None:
    if _ACTIVE_JOB_ID and _ACTIVE_JOB_ID in _JOBS:
        return _JOBS[_ACTIVE_JOB_ID]
    return None


def get_status() -> AnalysisStatus:
    return AnalysisStatus(last_run_at=_LAST_RUN_AT, last_success_at=_LAST_SUCCESS_AT,
                          last_error=_LAST_ERROR, is_running=_ACTIVE_JOB_ID is not None)


def get_job(job_id: str) -> AnalysisJob | None:
    return _JOBS.get(job_id)


def _load_picks() -> tuple[str, list[tuple[str, str, dict]]]:
    """Derive (as_of_date, [(symbol, name, context), ...]) from the served candidates.
    Runs inside the worker thread (candidates() is sync + qlib-heavy)."""
    from app.models import service as models_service
    result = models_service.candidates()
    # candidates() returns items as dicts (ScreenItem.model_dump()) — use dict access.
    items = result["items"][: _top_n(Settings())]
    as_of = result.get("as_of_date") or result.get("latest_date") or ""
    picks = [
        (it["symbol"], it.get("name", ""), {
            "score_today": it.get("score_today"),
            "pct_change_5d": it.get("pct_change_5d"),
            "board": it.get("board"),
            "is_st": it.get("is_st"),
        })
        for it in items
    ]
    return as_of, picks


def _analyze_symbol(symbol: str, name: str, ctx: dict, as_of: str):
    """Fetch + LLM for one pick. Returns AiAnalysis or None on hard error (2 attempts).
    Provider + model come from Settings (openai / deepseek / anthropic)."""
    kind, client, model = make_client(Settings())
    news = fetch_news(symbol)
    notices = fetch_notices(symbol)
    for attempt in (1, 2):
        try:
            return analyze_one(kind, client, model, symbol=symbol, name=name,
                               news=news, notices=notices, context=ctx, as_of_date=as_of)
        except Exception as exc:
            log.warning("analyze_failed symbol=%s attempt=%d: %s", symbol, attempt, exc)
    return None


def _run_picks(job_id: str, db_path: str) -> int:
    """Worker body — overridable in tests. Returns count analyzed."""
    as_of, picks = _load_picks()
    if not picks:
        return 0
    rows: list = []
    with ThreadPoolExecutor(max_workers=_CONCURRENCY) as ex:
        fut_to_sym = {
            ex.submit(_analyze_symbol, sym, name, ctx, as_of): sym
            for sym, name, ctx in picks
        }
        for fut, sym in fut_to_sym.items():
            a = fut.result()
            if a is not None:
                rows.append((sym, a))
    store.upsert_many(db_path, rows)
    with _LOCK:
        j = _JOBS.get(job_id)
        if j:
            j.as_of_date = as_of
    return len(rows)


def trigger_analysis(reason: str = "manual_ui") -> TriggerResponse:
    global _ACTIVE_JOB_ID, _LAST_RUN_AT
    s = Settings()
    if not _is_enabled(s):
        return TriggerResponse(status="disabled", job_id=None)

    with _LOCK:
        if _ACTIVE_JOB_ID and _ACTIVE_JOB_ID in _JOBS \
           and _JOBS[_ACTIVE_JOB_ID].status == "running":
            return TriggerResponse(status="already_running", job_id=_ACTIVE_JOB_ID)
        job_id = uuid.uuid4().hex[:12]
        now = datetime.utcnow().isoformat()
        _remember(job_id, AnalysisJob(job_id=job_id, status="running",
                                      started_at=now, reason=reason))
        _ACTIVE_JOB_ID = job_id
        _LAST_RUN_AT = now

    db_path = _db_path(s)
    threading.Thread(target=_worker, args=(job_id, db_path), daemon=True).start()
    return TriggerResponse(status="started", job_id=job_id)


def _worker(job_id: str, db_path: str) -> None:
    global _ACTIVE_JOB_ID, _LAST_SUCCESS_AT, _LAST_ERROR
    analyzed = None
    err = None
    try:
        analyzed = _run_picks(job_id, db_path)
    except Exception as exc:
        log.exception("analysis_worker_error job_id=%s: %s", job_id, exc)
        err = str(exc)[-2000:]
    finally:
        with _LOCK:
            j = _JOBS.get(job_id)
            if j:
                j.status = "failed" if err else "done"
                j.finished_at = datetime.utcnow().isoformat()
                j.analyzed = analyzed
                if err:
                    j.error = err
                    _LAST_ERROR = err
                else:
                    _LAST_SUCCESS_AT = j.finished_at
                    _LAST_ERROR = None
            _ACTIVE_JOB_ID = None
