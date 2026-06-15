import json

from production.progress import emit_progress


def test_emit_progress_prints_parseable_progress_line(capsys):
    emit_progress("train", 3, 9, "training lgbm")
    out = capsys.readouterr().out.strip()
    assert out.startswith("PROGRESS ")
    payload = json.loads(out[len("PROGRESS "):])
    assert payload == {"phase": "train", "current": 3, "total": 9, "message": "training lgbm"}


def test_emit_progress_defaults_empty_message(capsys):
    emit_progress("done", 9, 9)
    out = capsys.readouterr().out.strip()
    payload = json.loads(out[len("PROGRESS "):])
    assert payload["message"] == ""
    assert payload["phase"] == "done"
