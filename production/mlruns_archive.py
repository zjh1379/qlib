"""Auto-archive mlruns recorders older than N weeks.

Per spec section R5: mlruns directory grows unbounded; move recorders > 8 weeks old
to production/archive/<exp_id>/<recorder_id>/.
"""
from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path

_log = logging.getLogger("mlruns_archive")


def _recorder_age_weeks(timestamp: datetime, now: datetime | None = None) -> float:
    now = now or datetime.now()
    return (now - timestamp).total_seconds() / (7 * 24 * 3600)


def archive_old_recorders(
    mlruns_root: Path,
    archive_root: Path,
    keep_weeks: int = 8,
    now: datetime | None = None,
) -> int:
    """Walk `mlruns_root/<exp_id>/<recorder_id>/` and move any recorder with
    `meta.yaml` mtime older than `keep_weeks` into `archive_root/<exp_id>/<recorder_id>/`.

    Returns the number of archived recorders.
    """
    now = now or datetime.now()
    moved = 0
    if not mlruns_root.exists():
        return 0
    for exp_dir in mlruns_root.iterdir():
        if not exp_dir.is_dir():
            continue
        for rec_dir in exp_dir.iterdir():
            if not rec_dir.is_dir():
                continue
            meta = rec_dir / "meta.yaml"
            if not meta.exists():
                continue
            ts = datetime.fromtimestamp(rec_dir.stat().st_mtime)
            age_weeks = _recorder_age_weeks(ts, now=now)
            if age_weeks > keep_weeks:
                dest = archive_root / exp_dir.name / rec_dir.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(rec_dir), str(dest))
                moved += 1
                _log.info("recorder_archived dest=%s age_weeks=%s", str(dest), age_weeks)
    return moved
