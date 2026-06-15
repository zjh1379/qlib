"""Resumable-refresh checkpoint helpers (root-caused 2026-06-15: stuck refresh
restarted from scratch after interruption)."""
import production.incremental_refresh as ir


def test_checkpoint_roundtrip_and_prunes_old_days(tmp_path):
    cycle = "2026-06-15"
    # An old-day checkpoint should be pruned the first time we read today's.
    (tmp_path / ".refresh_progress_2026-06-01.txt").write_text("sh.000001\n", encoding="utf-8")

    assert ir._read_done_set(tmp_path, cycle) == set()            # nothing done today yet
    assert not (tmp_path / ".refresh_progress_2026-06-01.txt").exists()  # old day pruned

    ir._mark_done(tmp_path, cycle, "sh.600000")
    ir._mark_done(tmp_path, cycle, "sh.600009")
    assert ir._read_done_set(tmp_path, cycle) == {"sh.600000", "sh.600009"}

    # today's checkpoint is kept across reads (so same-day re-runs stay resumable)
    assert (tmp_path / f".refresh_progress_{cycle}.txt").exists()
