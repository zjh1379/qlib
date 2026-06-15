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
    s = Settings(_env_file=None)              # ignore any local .env; test true code defaults
    assert s.ai_provider == "openai"          # default provider (OpenAI; DeepSeek via same SDK)
    assert s.ai_model == ""                    # blank = per-provider default
    assert s.ai_analysis_top_n == 10
    assert s.ai_analysis_enabled is False
    assert s.openai_api_key == "" and s.deepseek_api_key == "" and s.anthropic_api_key == ""

    monkeypatch.setenv("QLIB_COMPANION_AI_PROVIDER", "deepseek")
    monkeypatch.setenv("QLIB_COMPANION_DEEPSEEK_API_KEY", "sk-ds")
    monkeypatch.setenv("QLIB_COMPANION_AI_ANALYSIS_ENABLED", "true")
    monkeypatch.setenv("QLIB_COMPANION_AI_MODEL", "deepseek-chat")
    s2 = Settings(_env_file=None)
    assert s2.ai_provider == "deepseek"
    assert s2.deepseek_api_key == "sk-ds"
    assert s2.ai_analysis_enabled is True
    assert s2.ai_model == "deepseek-chat"
