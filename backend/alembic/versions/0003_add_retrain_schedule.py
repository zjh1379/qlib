"""add retrain_schedule table

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-21

Single-row config for the weekly retrain cron. Seeded with default
Sunday 22:00 on first upgrade.
"""
from alembic import op
import sqlalchemy as sa


revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "retrain_schedule",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("day_of_week", sa.Integer(), nullable=False),  # 0=Mon … 6=Sun
        sa.Column("hour", sa.Integer(), nullable=False),
        sa.Column("minute", sa.Integer(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("last_run_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("next_run_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.current_timestamp(),
        ),
        sa.CheckConstraint("day_of_week BETWEEN 0 AND 6", name="ck_retrain_schedule_dow_range"),
        sa.CheckConstraint("hour BETWEEN 0 AND 23", name="ck_retrain_schedule_hour_range"),
        sa.CheckConstraint("minute BETWEEN 0 AND 59", name="ck_retrain_schedule_minute_range"),
        sa.CheckConstraint("id = 1", name="ck_retrain_schedule_single_row"),
    )
    # Seed default row
    op.execute(
        "INSERT OR IGNORE INTO retrain_schedule (id, day_of_week, hour, minute, enabled) "
        "VALUES (1, 6, 22, 0, 1)"
    )


def downgrade() -> None:
    op.drop_table("retrain_schedule")
