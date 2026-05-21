"""Tests for service._latest_progress: parsing PROGRESS lines from a refresh log."""
from __future__ import annotations

from pathlib import Path

from app.data.service import _latest_progress


def test_latest_progress_parses(tmp_path: Path):
    log = tmp_path / "x.log"
    log.write_text(
        "some noise\n"
        'PROGRESS {"phase":"fetch","current":10,"total":300,"message":"sh.600519"}\n'
        "more noise\n"
        'PROGRESS {"phase":"fetch","current":42,"total":300,"message":"sh.600036: +2 rows"}\n',
        encoding="utf-8",
    )
    p = _latest_progress(log)
    assert p is not None
    assert p.phase == "fetch"
    assert p.current == 42
    assert p.total == 300
    assert "600036" in p.message


def test_latest_progress_missing_returns_none(tmp_path: Path):
    assert _latest_progress(tmp_path / "absent.log") is None


def test_latest_progress_no_progress_lines(tmp_path: Path):
    log = tmp_path / "x.log"
    log.write_text("just some plain output\nno structured progress\n", encoding="utf-8")
    assert _latest_progress(log) is None


def test_latest_progress_handles_malformed_then_falls_back_to_earlier(tmp_path: Path):
    """If the most recent PROGRESS line has invalid JSON, fall back to the previous valid one."""
    log = tmp_path / "x.log"
    log.write_text(
        'PROGRESS {"phase":"init","current":1,"total":1,"message":"hi"}\n'
        "PROGRESS not-json-garbage\n",
        encoding="utf-8",
    )
    p = _latest_progress(log)
    assert p is not None
    assert p.phase == "init"
    assert p.message == "hi"


def test_latest_progress_handles_non_utf8_bytes(tmp_path: Path):
    """The log file may contain non-UTF8 bytes from baostock errors; we must not crash."""
    log = tmp_path / "x.log"
    # Mix of invalid bytes + a valid PROGRESS line afterwards
    log.write_bytes(
        b"\xff\xfe garbage bytes here\n"
        b'PROGRESS {"phase":"done","current":1,"total":1,"message":"ok"}\n'
    )
    p = _latest_progress(log)
    assert p is not None
    assert p.phase == "done"
