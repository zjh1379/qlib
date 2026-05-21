from sqlalchemy import CheckConstraint, Column, DateTime, Float, Index, Integer, String
from sqlalchemy.sql import func

from app.core.db import Base


class TransactionORM(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False, index=True)
    kind = Column(String, nullable=False)
    qty = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    fee = Column(Float, default=0, nullable=False)
    executed_at = Column(DateTime, nullable=False)
    broker = Column(String, nullable=True)
    notes = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.current_timestamp(), nullable=False)

    __table_args__ = (
        CheckConstraint("kind IN ('buy','sell')", name="ck_transactions_kind"),
        CheckConstraint("qty > 0", name="ck_transactions_qty_pos"),
        CheckConstraint("price > 0", name="ck_transactions_price_pos"),
        Index("idx_tx_symbol_time", "symbol", "executed_at"),
    )
