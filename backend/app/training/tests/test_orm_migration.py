import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import create_engine

from app.training.orm import TrainingRunORM
from app.core.db import Base


def test_training_run_orm_creates_and_roundtrips(tmp_path: Path):
    from sqlalchemy.orm import Session
    eng = create_engine(f"sqlite:///{tmp_path/'t.db'}")
    Base.metadata.create_all(eng, tables=[TrainingRunORM.__table__])
    with Session(eng) as s:
        s.add(TrainingRunORM(
            job_id="j1", kind="manual", scope="full", models_json=None,
            status="running", started_at="2026-06-16T01:00:00", finished_at=None,
            recorder_id=None, error=None,
        ))
        s.commit()
        row = s.get(TrainingRunORM, "j1")
        assert row.status == "running"
        assert row.scope == "full"
        assert row.recorder_id is None


def test_table_name_is_training_runs():
    assert TrainingRunORM.__tablename__ == "training_runs"
