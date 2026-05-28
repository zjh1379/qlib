"""Pydantic schemas for the inference module."""
from pydantic import BaseModel


class InferenceJob(BaseModel):
    job_id: str
    status: str  # "running" | "done" | "failed"
    started_at: str
    finished_at: str | None = None
    end_date: str | None = None
    error: str | None = None
    new_rows: int | None = None  # populated on success
    reason: str | None = None  # "manual_ui" | "data_refresh" | ...


class InferenceStatus(BaseModel):
    last_run_at: str | None = None
    last_success_at: str | None = None
    last_error: str | None = None
    is_running: bool = False


class TriggerResponse(BaseModel):
    status: str  # "started" | "already_running"
    job_id: str | None = None
