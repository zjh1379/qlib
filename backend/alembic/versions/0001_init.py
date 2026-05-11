"""init

Revision ID: 0001
Revises:
Create Date: 2026-05-11

This migration intentionally creates no tables.
P1 has no SQLite-backed state; tables are added in P2 (portfolio) onward.
"""
from alembic import op  # noqa: F401
import sqlalchemy as sa  # noqa: F401


revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
