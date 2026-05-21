from pathlib import Path

import pytest


@pytest.fixture
def tmp_cache_path(tmp_path: Path) -> Path:
    return tmp_path / "pit_constituents.parquet"
