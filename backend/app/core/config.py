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
    default_experiment: str = "daily_cn_fresh"
    default_chart_window_days: int = 365

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
