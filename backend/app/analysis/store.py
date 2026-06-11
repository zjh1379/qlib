"""ai_analysis persistence. Writes are sync sqlite3 (worker thread); reads are
async ORM (serving path). Same table, created by alembic migration 0004."""
from __future__ import annotations

import json
import logging
import sqlite3

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.analysis.orm import AiAnalysisORM
from app.analysis.schemas import AiAnalysis, RiskFlag

log = logging.getLogger(__name__)


def upsert_many(db_path: str, rows: list[tuple[str, AiAnalysis]]) -> int:
    """INSERT OR REPLACE (symbol, AiAnalysis) pairs. Synchronous — call from the worker thread."""
    if not rows:
        return 0
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        conn.executemany(
            """INSERT OR REPLACE INTO ai_analysis
               (symbol, as_of_date, interpretation, risk_flags_json, stance, model, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
            [
                (
                    sym, a.as_of_date, a.interpretation,
                    json.dumps([f.model_dump() for f in a.risk_flags], ensure_ascii=False),
                    a.stance, a.model, a.status,
                )
                for sym, a in rows
            ],
        )
        conn.commit()
        return len(rows)
    finally:
        conn.close()


async def fetch_analyses(
    session: AsyncSession, symbols: list[str], as_of_date: str,
) -> dict[str, AiAnalysis]:
    """Read non-failed analyses for these symbols at as_of_date. Keyed by symbol."""
    if not symbols or not as_of_date:
        return {}
    res = await session.execute(
        select(AiAnalysisORM).where(
            AiAnalysisORM.symbol.in_(symbols),
            AiAnalysisORM.as_of_date == as_of_date,
            AiAnalysisORM.status != "failed",
        )
    )
    out: dict[str, AiAnalysis] = {}
    for row in res.scalars().all():
        try:
            flags = [RiskFlag(**f) for f in json.loads(row.risk_flags_json or "[]")]
        except Exception:
            flags = []
        out[row.symbol] = AiAnalysis(
            interpretation=row.interpretation, risk_flags=flags, stance=row.stance,
            model=row.model, as_of_date=row.as_of_date, status=row.status,
        )
    return out
