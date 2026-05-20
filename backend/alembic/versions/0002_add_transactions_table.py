"""add transactions table

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-20

P2 portfolio module: persistent ledger of buy/sell transactions.
Holdings are derived from this table at read time (no materialized table).
"""
from alembic import op
import sqlalchemy as sa


revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "transactions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("symbol", sa.String(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("qty", sa.Float(), nullable=False),
        sa.Column("price", sa.Float(), nullable=False),
        sa.Column("fee", sa.Float(), nullable=False, server_default="0"),
        sa.Column("executed_at", sa.DateTime(), nullable=False),
        sa.Column("broker", sa.String(), nullable=True),
        sa.Column("notes", sa.String(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.current_timestamp(),
        ),
        sa.CheckConstraint("kind IN ('buy','sell')", name="ck_transactions_kind"),
        sa.CheckConstraint("qty > 0", name="ck_transactions_qty_pos"),
        sa.CheckConstraint("price > 0", name="ck_transactions_price_pos"),
    )
    op.create_index(
        "ix_transactions_symbol", "transactions", ["symbol"], unique=False
    )
    op.create_index(
        "idx_tx_symbol_time", "transactions", ["symbol", "executed_at"], unique=False
    )


def downgrade() -> None:
    op.drop_index("idx_tx_symbol_time", table_name="transactions")
    op.drop_index("ix_transactions_symbol", table_name="transactions")
    op.drop_table("transactions")
