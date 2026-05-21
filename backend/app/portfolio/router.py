from datetime import datetime

from fastapi import APIRouter, Depends, Query

from app.core.db import get_session
from app.portfolio import service
from app.portfolio.schemas import (
    HoldingsResponse,
    Transaction,
    TransactionIn,
    TransactionUpdate,
)

router = APIRouter()


@router.get("/holdings", response_model=HoldingsResponse)
async def holdings(session=Depends(get_session)):
    return await service.get_holdings(session)


@router.get("/transactions", response_model=list[Transaction])
async def list_transactions(
    symbol: str | None = Query(None),
    from_date: datetime | None = Query(None, alias="from"),
    to_date: datetime | None = Query(None, alias="to"),
    session=Depends(get_session),
):
    return await service.list_transactions(session, symbol, from_date, to_date)


@router.post("/transactions", response_model=Transaction)
async def add_transaction(payload: TransactionIn, session=Depends(get_session)):
    return await service.add_transaction(session, payload)


@router.get("/transactions/{tx_id}", response_model=Transaction)
async def get_transaction(tx_id: int, session=Depends(get_session)):
    return await service.get_transaction(session, tx_id)


@router.patch("/transactions/{tx_id}", response_model=Transaction)
async def update_transaction(
    tx_id: int, payload: TransactionUpdate, session=Depends(get_session)
):
    return await service.update_transaction(session, tx_id, payload)


@router.delete("/transactions/{tx_id}", status_code=204)
async def delete_transaction(tx_id: int, session=Depends(get_session)):
    await service.delete_transaction(session, tx_id)
    return None
