from sqlalchemy import Boolean, CheckConstraint, Column, DateTime, Integer
from sqlalchemy.sql import func

from app.core.db import Base


class RetrainScheduleORM(Base):
    __tablename__ = "retrain_schedule"

    id = Column(Integer, primary_key=True)
    day_of_week = Column(Integer, nullable=False)
    hour = Column(Integer, nullable=False)
    minute = Column(Integer, nullable=False)
    enabled = Column(Boolean, nullable=False, default=True)
    last_run_at = Column(DateTime(timezone=True), nullable=True)
    next_run_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.current_timestamp(), nullable=False)

    __table_args__ = (
        CheckConstraint("day_of_week BETWEEN 0 AND 6", name="ck_retrain_schedule_dow_range"),
        CheckConstraint("hour BETWEEN 0 AND 23", name="ck_retrain_schedule_hour_range"),
        CheckConstraint("minute BETWEEN 0 AND 59", name="ck_retrain_schedule_minute_range"),
        CheckConstraint("id = 1", name="ck_retrain_schedule_single_row"),
    )
