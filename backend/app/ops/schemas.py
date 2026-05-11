from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str
    qlib_ready: bool
    calendar_end: str | None = None
