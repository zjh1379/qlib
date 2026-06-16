from pydantic import BaseModel, Field


class TrainingProgress(BaseModel):
    phase: str = Field(..., description='"universe" | "train" | "ensemble" | "done" (or future phases)')
    current: int = Field(..., description="Current step (1-based); equals total when finished")
    total: int = Field(..., description="Total steps for this run")
    message: str = Field("", description="Human-readable status line, e.g. 'training lgbm'")


class TrainRequest(BaseModel):
    # P1: full ensemble only. `scope`/`models` reserved for P3 (single-algo).
    scope: str = Field("full", description='"full" (P1). Single-algo arrives in P3.')
    force: bool = Field(False, description="Override the trading-hours guard.")


class TrainingJobStatus(BaseModel):
    job_id: str
    kind: str = Field(..., description='"cron" | "manual"')
    status: str = Field(..., description='"pending" | "running" | "done" | "failed" | "skipped"')
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    progress: TrainingProgress | None = None
    log_tail: str | None = None
