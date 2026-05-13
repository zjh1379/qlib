from fastapi import APIRouter, Query

from app.core.exceptions import BusinessError
from app.data.schemas import (
    DataStatus,
    InstrumentsResponse,
    RefreshJobStatus,
    RefreshResponse,
)
from app.data.service import (
    get_csi300,
    get_data_status,
    get_refresh_status,
    start_refresh,
)

# Router for /api/data/* endpoints (mounted with prefix="/api/data")
router = APIRouter()

# Separate router for /api/instruments (mounted with prefix="/api")
instruments_router = APIRouter()


@router.get("/status", response_model=DataStatus)
def data_status() -> DataStatus:
    return get_data_status()


@router.post("/refresh", response_model=RefreshResponse)
def refresh() -> RefreshResponse:
    return start_refresh()


@router.get("/refresh/{job_id}", response_model=RefreshJobStatus)
def refresh_status(job_id: str) -> RefreshJobStatus:
    return get_refresh_status(job_id)


@instruments_router.get("/instruments", response_model=InstrumentsResponse)
def instruments(
    market: str = Query(default="csi300", description="Market identifier, only csi300 supported"),
) -> InstrumentsResponse:
    if market != "csi300":
        raise BusinessError(
            "market not supported",
            code="unsupported_market",
            context={"market": market},
        )
    return get_csi300()
