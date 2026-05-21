from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from production import pit_constituents as pit


def _mk_baostock_df(n_rows: int, prefix: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": [f"{prefix}.{i:06d}" for i in range(n_rows)],
            "code_name": [f"name_{i}" for i in range(n_rows)],
        }
    )


def test_sanity_check_ranges():
    csi300 = _mk_baostock_df(300, "sh")
    csi500 = _mk_baostock_df(500, "sh")
    assert pit._is_within_range(len(csi300), pit.CSI300_RANGE)
    assert pit._is_within_range(len(csi500), pit.CSI500_RANGE)


def test_sanity_check_rejects_undersized():
    assert not pit._is_within_range(100, pit.CSI300_RANGE)


def test_fetch_uses_cache_when_fresh(tmp_cache_path):
    cached = pd.DataFrame(
        {
            "snapshot_date": [date(2024, 1, 1), date(2024, 2, 1)],
            "instrument": ["SH600000", "SH600001"],
            "membership": ["csi300", "csi500"],
        }
    )
    cached.to_parquet(tmp_cache_path)

    with patch.object(pit, "_fetch_remote") as mock_remote:
        result = pit.load_or_refresh(end=date(2024, 2, 1), cache_path=tmp_cache_path, allow_stale_days=120)
        mock_remote.assert_not_called()

    assert len(result) == 2


def test_fetch_remote_when_stale(tmp_cache_path):
    cached = pd.DataFrame(
        {
            "snapshot_date": [date(2020, 1, 1)],
            "instrument": ["SH600000"],
            "membership": ["csi300"],
        }
    )
    cached.to_parquet(tmp_cache_path)

    fresh = pd.DataFrame(
        {
            "snapshot_date": [date(2026, 5, 1)] * 800,
            "instrument": [f"SH{600000+i}" for i in range(800)],
            "membership": ["csi300"] * 300 + ["csi500"] * 500,
        }
    )
    with patch.object(pit, "_fetch_remote", return_value=fresh) as mock_remote:
        result = pit.load_or_refresh(end=date(2026, 5, 1), cache_path=tmp_cache_path, allow_stale_days=30)
        mock_remote.assert_called_once()

    assert len(result) == 800


def test_pit_lookup_returns_membership_for_date():
    df = pd.DataFrame(
        {
            "snapshot_date": pd.to_datetime([date(2024, 1, 1)] * 3 + [date(2024, 2, 1)] * 3),
            "instrument": ["SH600000", "SH600001", "SH600002", "SH600000", "SH600001", "SH600002"],
            "membership": ["csi300", "csi500", "csi300", "csi300", "csi300", "csi500"],
        }
    )
    # Query for 2024-01-15 -> should use 2024-01-01 snapshot
    members = pit.members_on(df, date(2024, 1, 15))
    assert set(members) == {"SH600000", "SH600001", "SH600002"}

    # Query for 2024-02-10 -> should use 2024-02-01 snapshot
    members = pit.members_on(df, date(2024, 2, 10))
    assert set(members) == {"SH600000", "SH600001", "SH600002"}


def test_write_pit_instruments_file_format(tmp_path: Path):
    df = pd.DataFrame(
        {
            "snapshot_date": pd.to_datetime([date(2024, 1, 1)] * 2 + [date(2024, 2, 1)] * 2),
            "instrument": ["SH600000", "SH600001", "SH600000", "SH600002"],
            "membership": ["csi300", "csi300", "csi300", "csi500"],
        }
    )
    qlib_root = tmp_path / "qlib_data" / "cn_data_bs"
    out_path = pit.write_pit_instruments_file(
        df,
        end_date=date(2024, 2, 1),
        name="csi800_pit_test",
        qlib_data_root=qlib_root,
        lookback_years=1,
    )

    # File exists at <qlib_root>/instruments/<name>.txt
    assert out_path == qlib_root / "instruments" / "csi800_pit_test.txt"
    assert out_path.exists()

    # TSV format: instrument <TAB> start_date <TAB> end_date
    lines = out_path.read_text().strip().splitlines()
    # Union of all members across both snapshots = SH600000, SH600001, SH600002
    parsed = sorted(line.split("\t")[0] for line in lines)
    assert parsed == ["SH600000", "SH600001", "SH600002"]
    # Every line has 3 tab-separated columns
    for line in lines:
        parts = line.split("\t")
        assert len(parts) == 3


def test_fetch_fallback_to_cache_on_remote_error(tmp_cache_path):
    """If _fetch_remote raises, load_or_refresh returns the cached df rather
    than propagating the error — even when the cache is stale beyond allow_stale_days."""
    cached = pd.DataFrame(
        {
            "snapshot_date": [date(2020, 1, 1)],
            "instrument": ["SH600000"],
            "membership": ["csi300"],
        }
    )
    cached.to_parquet(tmp_cache_path)
    with patch.object(pit, "_fetch_remote", side_effect=RuntimeError("network down")):
        result = pit.load_or_refresh(
            end=date(2026, 5, 1), cache_path=tmp_cache_path, allow_stale_days=30
        )
    assert len(result) == 1
    assert result.iloc[0]["instrument"] == "SH600000"
