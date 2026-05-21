from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class TransactionIn(BaseModel):
    symbol: str = Field(..., pattern=r"^(SH|SZ)\d{6}$")
    kind: Literal["buy", "sell"]
    qty: float = Field(..., gt=0)
    price: float = Field(..., gt=0)
    fee: float = Field(default=0, ge=0)
    executed_at: datetime
    broker: str | None = None
    notes: str | None = None


class TransactionUpdate(BaseModel):
    qty: float | None = Field(default=None, gt=0)
    price: float | None = Field(default=None, gt=0)
    fee: float | None = Field(default=None, ge=0)
    executed_at: datetime | None = None
    broker: str | None = None
    notes: str | None = None


class Transaction(BaseModel):
    id: int
    symbol: str
    kind: Literal["buy", "sell"]
    qty: float
    price: float
    fee: float
    executed_at: datetime
    broker: str | None = None
    notes: str | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class Holding(BaseModel):
    symbol: str
    name: str = ""                   # populated from qlib_adapter
    qty: float
    avg_cost: float                  # weighted-average buy price including fees
    effective_cost: float            # (cost_in - cash_out) / qty — accounts for realized P&L
    current_price: float | None = None
    market_value: float | None = None
    unrealized_pnl: float | None = None
    unrealized_pnl_pct: float | None = None


class HoldingsResponse(BaseModel):
    holdings: list[Holding]
    total_cost: float
    total_market_value: float | None = None
    total_unrealized_pnl: float | None = None
    as_of: str | None = None         # calendar_end date (when current_price was sourced)
