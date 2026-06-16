from pathlib import Path

from app.training.service import latest_progress, tail_log, build_job_status


def test_latest_progress_parses_last_progress_line(tmp_path: Path):
    log = tmp_path / "j.log"
    log.write_text(
        'PROGRESS {"phase":"universe","current":1,"total":6,"message":"u"}\n'
        'some noise\n'
        'PROGRESS {"phase":"train","current":2,"total":6,"message":"training lgbm"}\n',
        encoding="utf-8",
    )
    p = latest_progress(log)
    assert p is not None
    assert p.phase == "train"
    assert p.current == 2
    assert p.total == 6
    assert p.message == "training lgbm"


def test_latest_progress_missing_or_empty_returns_none(tmp_path: Path):
    assert latest_progress(tmp_path / "absent.log") is None
    empty = tmp_path / "e.log"
    empty.write_text("plain output, no progress\n", encoding="utf-8")
    assert latest_progress(empty) is None


def test_build_job_status_enriches_entry_with_progress(tmp_path: Path):
    log = tmp_path / "j.log"
    log.write_text('PROGRESS {"phase":"done","current":6,"total":6,"message":"done"}\n', encoding="utf-8")
    entry = {
        "job_id": "j1", "kind": "manual", "status": "done",
        "started_at": "s", "finished_at": "f", "error": None, "log_path": str(log),
    }
    st = build_job_status(entry)
    assert st.job_id == "j1"
    assert st.status == "done"
    assert st.progress is not None and st.progress.phase == "done"
    assert "PROGRESS" in (st.log_tail or "")


def test_build_job_status_handles_missing_log_path(tmp_path: Path):
    entry = {
        "job_id": "j2", "kind": "manual", "status": "pending",
        "started_at": None, "finished_at": None, "error": None, "log_path": None,
    }
    st = build_job_status(entry)
    assert st.progress is None
    assert st.log_tail is None
