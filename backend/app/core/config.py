from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="QLIB_COMPANION_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Storage paths
    qlib_provider_uri: str = "~/.qlib/qlib_data/cn_data_bs"
    qlib_region: str = "cn"
    mlruns_dir: str = "examples/mlruns"
    app_db_path: str = "app.db"

    # Defaults
    # β phase switchover 2026-05-26: LGBM × 3-horizon rank-avg ensemble
    # beats daily_cn_fresh on IC (0.0349 vs 0.0263), MDD (-9.3% vs -20.4%),
    # turnover (32% vs 49%) over the same 53 trading day evaluation window.
    # See production/reports/backfill_eval_20260526_090328.json for details.
    default_experiment: str = "rolling_v2_ensemble"
    default_chart_window_days: int = 365

    # Retrain / rolling-ensemble subprocess settings
    retrain_recorder_experiment: str = "rolling_v2_ensemble"
    retrain_python_path: str = "F:/Tools/Anaconda/envs/qlib/python.exe"

    # AI analysis layer (解读 + 风险旗标)
    ai_provider: str = "openai"           # openai | deepseek | anthropic
    openai_api_key: str = ""              # QLIB_COMPANION_OPENAI_API_KEY
    deepseek_api_key: str = ""            # QLIB_COMPANION_DEEPSEEK_API_KEY
    anthropic_api_key: str = ""           # QLIB_COMPANION_ANTHROPIC_API_KEY
    ai_model: str = ""                    # blank = per-provider default (openai=gpt-4o-mini, deepseek=deepseek-chat, anthropic=claude-sonnet-4-6)
    ai_analysis_top_n: int = 10           # analyze the top-N picks per run
    ai_analysis_enabled: bool = False     # off until a provider key is set

    @property
    def db_url(self) -> str:
        path = Path(self.app_db_path).expanduser().resolve()
        return f"sqlite+aiosqlite:///{path}"

    @property
    def qlib_data_dir(self) -> Path:
        return Path(self.qlib_provider_uri).expanduser().resolve()

    @property
    def mlruns_path(self) -> Path:
        """Resolve mlruns_dir against the project root (3 levels up from this file:
        config.py -> core/ -> app/ -> backend/ -> <repo root>).
        If mlruns_dir is absolute, use it as-is.
        """
        p = Path(self.mlruns_dir).expanduser()
        if p.is_absolute():
            return p.resolve()
        # config.py is at backend/app/core/config.py -> parents[3] is the worktree root
        project_root = Path(__file__).resolve().parents[3]
        return (project_root / p).resolve()
