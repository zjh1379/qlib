from pydantic import BaseModel, Field


class ProgressInfo(BaseModel):
    phase: str = Field(
        ...,
        description='One of "init" | "fetch" | "dump" | "benchmark" | "done" (or future phases)',
    )
    current: int = Field(..., description="Current step (0-indexed start, total when finished)")
    total: int = Field(..., description="Total steps for this phase")
    message: str = Field("", description="Human-readable status line for this step")


class DataStatus(BaseModel):
    calendar_end: str = Field(..., description="ISO date YYYY-MM-DD of the last trading day")
    calendar_size: int = Field(..., description="Number of trading days in the calendar")
    instruments_count: int = Field(..., description="Number of symbols in csi300 (~300)")
    last_refresh_at: str | None = Field(
        None, description="ISO UTC timestamp; mtime of calendars/day.txt"
    )
    freshness: str = Field(
        ..., description='One of "fresh" | "stale_1d" | "stale_2d_plus"'
    )


class InstrumentItem(BaseModel):
    symbol: str = Field(..., description='e.g. "SH600519"')
    name: str = Field(..., description='Chinese name, "" if unknown')


class InstrumentsResponse(BaseModel):
    market: str
    count: int
    items: list[InstrumentItem]


class RefreshResponse(BaseModel):
    job_id: str
    started_at: str
    message: str


class RefreshJobStatus(BaseModel):
    job_id: str
    status: str = Field(..., description='"running" | "done" | "failed"')
    started_at: str
    finished_at: str | None = None
    log_tail: str | None = None
    progress: ProgressInfo | None = Field(
        None,
        description="Latest structured progress emitted by the refresh script, if any.",
    )


class MarketInfo(BaseModel):
    name: str = Field(..., description='"csi300" | "csi500" | "etfs" | "custom" | ...')
    label: str = Field(..., description='Human-readable label, e.g. "沪深300"')
    count: int = Field(..., description="Number of symbols in this market")


class MarketsResponse(BaseModel):
    markets: list[MarketInfo]
    total: int = Field(..., description="Sum of all markets' counts with dedup")


class AddSymbolRequest(BaseModel):
    symbol: str = Field(..., description='qlib format e.g. "SH601398"')


class AddSymbolResponse(BaseModel):
    symbol: str
    fetched_rows: int
    message: str
