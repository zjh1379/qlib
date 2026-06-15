"""get_calendar_info must read calendars/day.txt directly so the data-status
panel reflects a data refresh immediately, without a backend restart
(root-caused 2026-06-15: app showed 2026-06-02 while disk calendar was 2026-06-15
because qlib's D.calendar() was cached at init)."""
from datetime import date

from app.core import qlib_adapter


def test_get_calendar_info_reads_day_txt_fresh(tmp_path, monkeypatch):
    cal_dir = tmp_path / "calendars"
    cal_dir.mkdir(parents=True)
    (cal_dir / "day.txt").write_text(
        "2026-05-29\n2026-06-01\n2026-06-02\n2026-06-12\n2026-06-15\n", encoding="utf-8"
    )
    # Point the configured qlib data dir at our tmp dir.
    monkeypatch.setenv("QLIB_COMPANION_QLIB_PROVIDER_URI", str(tmp_path))

    last, total = qlib_adapter.get_calendar_info()
    assert last == date(2026, 6, 15)   # fresh from disk (not qlib's cached calendar)
    assert total == 5
