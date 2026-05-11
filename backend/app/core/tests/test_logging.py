import json
import logging
from app.core.logging import configure_logging, get_logger


def test_logger_emits_json(capsys):
    configure_logging(json_output=True)
    log = get_logger("test")
    log.info("hello", count=3)
    captured = capsys.readouterr().out
    data = json.loads(captured.strip().splitlines()[-1])
    assert data["event"] == "hello"
    assert data["count"] == 3
    assert data["level"] == "info"


def test_logger_emits_console(capsys):
    configure_logging(json_output=False)
    log = get_logger("test")
    log.warning("uh-oh", reason="x")
    out = capsys.readouterr().out
    assert "uh-oh" in out


def test_root_logger_respects_level():
    configure_logging(level="ERROR", json_output=True)
    assert logging.getLogger().level == logging.ERROR
