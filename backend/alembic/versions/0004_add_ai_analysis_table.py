"""add ai_analysis table

Revision ID: 0004
Revises: 0003
Create Date: 2026-06-11

AI analysis layer: stores per-symbol per-date LLM interpretation,
risk flags, stance, and generation metadata.
"""
from alembic import op
import sqlalchemy as sa

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ai_analysis",
        sa.Column("symbol", sa.String(), nullable=False),
        sa.Column("as_of_date", sa.String(), nullable=False),
        sa.Column("interpretation", sa.String(), nullable=False, server_default=""),
        sa.Column("risk_flags_json", sa.String(), nullable=False, server_default="[]"),
        sa.Column("stance", sa.String(), nullable=False, server_default="neutral"),
        sa.Column("model", sa.String(), nullable=False, server_default=""),
        sa.Column("status", sa.String(), nullable=False, server_default="ok"),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.current_timestamp(),
        ),
        sa.PrimaryKeyConstraint("symbol", "as_of_date"),
    )


def downgrade() -> None:
    op.drop_table("ai_analysis")
