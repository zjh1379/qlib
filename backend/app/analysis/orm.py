from sqlalchemy import Column, DateTime, String
from sqlalchemy.sql import func

from app.core.db import Base


class AiAnalysisORM(Base):
    __tablename__ = "ai_analysis"

    symbol = Column(String, primary_key=True)
    as_of_date = Column(String, primary_key=True)   # ISO date the picks are as-of
    interpretation = Column(String, nullable=False, default="")
    risk_flags_json = Column(String, nullable=False, default="[]")  # JSON list[RiskFlag]
    stance = Column(String, nullable=False, default="neutral")
    model = Column(String, nullable=False, default="")
    status = Column(String, nullable=False, default="ok")  # ok | partial | failed
    created_at = Column(DateTime, server_default=func.current_timestamp(), nullable=False)
