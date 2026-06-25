from pathlib import Path
from app.core.watchdog_supervisor import watchdog_cmd


def test_watchdog_cmd_builds_expected_args():
    cmd = watchdog_cmd(
        python_path="C:/py/python.exe",
        repo_root=Path("E:/Projects/qlib"),
        floor_gb=4.0,
        kill_pct=92.0,
        kills_path=Path("E:/Projects/qlib/logs/watchdog_kills.jsonl"),
    )
    assert cmd[0] == "C:/py/python.exe"
    assert cmd[1:4] == ["-m", "production.safety_watchdog", "--floor-gb"]
    assert "4.0" in cmd
    assert "--kill-pct" in cmd and "92.0" in cmd
    assert "--kills-path" in cmd
