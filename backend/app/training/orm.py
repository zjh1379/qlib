from sqlalchemy import Column, DateTime, String
from sqlalchemy.sql import func

from app.core.db import Base


class TrainingRunORM(Base):
    """One row per training run attempt (manual or cron). Populated by the
    SchedulerManager job lifecycle. recorder_id is filled on success once the
    training subprocess emits its `RECORDER <id>` line."""

    __tablename__ = "training_runs"

    job_id = Column(String, primary_key=True)
    kind = Column(String, nullable=False, default="manual")          # cron | manual
    scope = Column(String, nullable=False, default="full")           # full (P2) | single (P3)
    models_json = Column(String, nullable=True)                      # JSON list for single-algo (P3); null = all
    status = Column(String, nullable=False, default="pending")       # pending|running|done|failed|skipped
    started_at = Column(String, nullable=True)
    finished_at = Column(String, nullable=True)
    recorder_id = Column(String, nullable=True)
    error = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.current_timestamp(), nullable=False)
