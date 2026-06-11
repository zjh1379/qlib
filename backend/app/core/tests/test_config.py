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


def test_ai_analysis_settings_defaults_and_env(monkeypatch):
    from app.core.config import Settings
    s = Settings()
    assert s.ai_model == "claude-sonnet-4-6"   # spec default; user chose Sonnet for cost (2026-06-11)
    assert s.ai_analysis_top_n == 10
    assert s.ai_analysis_enabled is False
    assert s.anthropic_api_key == ""

    monkeypatch.setenv("QLIB_COMPANION_ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("QLIB_COMPANION_AI_ANALYSIS_ENABLED", "true")
    monkeypatch.setenv("QLIB_COMPANION_AI_MODEL", "claude-opus-4-8")   # override to a NON-default value proves env wins
    s2 = Settings()
    assert s2.anthropic_api_key == "sk-test"
    assert s2.ai_analysis_enabled is True
    assert s2.ai_model == "claude-opus-4-8"
