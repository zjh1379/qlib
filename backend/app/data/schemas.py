from pydantic import BaseModel, Field


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
