from datetime import datetime, timedelta
from pathlib import Path

import pytest

from production.mlruns_archive import archive_old_recorders, _recorder_age_weeks


def test_recorder_age_weeks_calculation():
    now = datetime(2026, 5, 21)
    older = datetime(2026, 3, 21)
    assert _recorder_age_weeks(older, now=now) == pytest.approx((61) / 7, rel=0.01)


def test_archive_moves_old_dirs(tmp_path: Path):
    src = tmp_path / "mlruns" / "1" / "abc123_old"
    src.mkdir(parents=True)
    (src / "meta.yaml").write_text("artifact_uri: foo\n")
    import os, time
    old_mtime = time.time() - 70 * 24 * 3600
    os.utime(src, (old_mtime, old_mtime))

    archive_dir = tmp_path / "archive"
    archive_old_recorders(mlruns_root=tmp_path / "mlruns", archive_root=archive_dir, keep_weeks=8)
    assert not src.exists()
    assert (archive_dir / "1" / "abc123_old").exists()


def test_archive_keeps_recent(tmp_path: Path):
    src = tmp_path / "mlruns" / "1" / "abc123_fresh"
    src.mkdir(parents=True)
    (src / "meta.yaml").write_text("artifact_uri: foo\n")
    archive_dir = tmp_path / "archive"
    archive_old_recorders(mlruns_root=tmp_path / "mlruns", archive_root=archive_dir, keep_weeks=8)
    assert src.exists()
    assert not (archive_dir / "1" / "abc123_fresh").exists()
