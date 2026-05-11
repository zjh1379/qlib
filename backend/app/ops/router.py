from fastapi import APIRouter

from app.core.qlib_adapter import get_calendar_end, init_qlib_once
from app.ops.schemas import HealthResponse

router = APIRouter()

APP_VERSION = "0.1.0"


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
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
