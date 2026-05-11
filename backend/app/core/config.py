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
    app_db_path: str = "backend/app.db"

    # Defaults
    default_experiment: str = "daily_cn_fresh"
    default_chart_window_days: int = 365

    @property
    def db_url(self) -> str:
        path = Path(self.app_db_path).expanduser().resolve()
        return f"sqlite+aiosqlite:///{path}"

    @property
    def qlib_data_dir(self) -> Path:
        return Path(self.qlib_provider_uri).expanduser().resolve()
