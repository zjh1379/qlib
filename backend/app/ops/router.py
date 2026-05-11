from fastapi import APIRouter

from app.core.qlib_adapter import _initialized as _qlib_initialized_flag  # noqa: F401

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    from app.core import qlib_adapter

    qlib_ready = bool(qlib_adapter._initialized)
    calendar_end: str | None = None
    if qlib_ready:
        try:
            calendar_end = qlib_adapter.get_calendar_end().isoformat()
        except Exception:
            calendar_end = None
    return {
        "status": "ok",
        "version": "0.1.0",
        "qlib_ready": qlib_ready,
        "calendar_end": calendar_end,
    }
