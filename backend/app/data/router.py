from fastapi import APIRouter, Query

from app.data.schemas import (
    AddSymbolRequest,
    AddSymbolResponse,
    DataStatus,
    InstrumentsResponse,
    MarketsResponse,
    RefreshJobStatus,
    RefreshResponse,
)
from app.data.service import (
    add_custom_symbol,
    get_data_status,
    get_refresh_status,
    list_instruments_for,
    list_markets,
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


@router.get("/markets", response_model=MarketsResponse)
def markets() -> MarketsResponse:
    return list_markets()


@router.post("/symbols/add", response_model=AddSymbolResponse)
def add_symbol(req: AddSymbolRequest) -> AddSymbolResponse:
    return add_custom_symbol(req.symbol)


@instruments_router.get("/instruments", response_model=InstrumentsResponse)
def instruments(
    market: str = Query(default="csi300", description="Market identifier; use 'all' for union"),
) -> InstrumentsResponse:
    return list_instruments_for(market)
