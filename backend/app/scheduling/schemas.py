from datetime import datetime

from pydantic import BaseModel, Field


class RetrainScheduleRead(BaseModel):
    day_of_week: int = Field(ge=0, le=6, description="0=Mon, 6=Sun")
    hour: int = Field(ge=0, le=23)
    minute: int = Field(ge=0, le=59)
    enabled: bool
    last_run_at: datetime | None
    next_run_at: datetime | None


class RetrainScheduleUpdate(BaseModel):
    day_of_week: int = Field(ge=0, le=6)
    hour: int = Field(ge=0, le=23)
    minute: int = Field(ge=0, le=59)
    enabled: bool


class RunNowResponse(BaseModel):
    status: str  # "started" | "rejected"
    reason: str | None = None
    job_id: str | None = None
