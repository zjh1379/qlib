from pathlib import Path
import os
from app.core.config import Settings


def test_settings_loads_defaults():
    s = Settings()
    assert s.api_port == 8000
    assert s.qlib_provider_uri.endswith("cn_data_bs")
    assert s.app_db_path.endswith("app.db")


def test_settings_overridable_by_env(monkeypatch):
    monkeypatch.setenv("QLIB_COMPANION_API_PORT", "9999")
    s = Settings()
    assert s.api_port == 9999


def test_resolved_paths_are_absolute():
    s = Settings()
    assert Path(s.qlib_provider_uri).expanduser().is_absolute()
