import pytest
import pandas as pd
from app.core.qlib_adapter import (
    init_qlib_once,
    get_ohlcv,
    get_calendar_end,
    get_csi300_instruments,
    load_pred,
    get_latest_recorder_id,
)


@pytest.fixture(scope="module", autouse=True)
def _init_qlib():
    init_qlib_once()


def test_get_csi300_returns_300ish():
    instruments = get_csi300_instruments()
    assert 200 <= len(instruments) <= 350
    assert all(s.startswith(("SH", "SZ")) for s in instruments)


def test_get_calendar_end_is_after_2025():
    end = get_calendar_end()
    assert end.year >= 2025


def test_get_ohlcv_returns_dataframe():
    end = get_calendar_end()
    df = get_ohlcv(["SH600519"], start="2025-01-01", end=str(end))
    assert isinstance(df, pd.DataFrame)
    assert {"$open", "$high", "$low", "$close", "$volume"}.issubset(df.columns)
    assert len(df) > 100


def test_load_pred_returns_series():
    rid = get_latest_recorder_id("daily_cn_fresh")
    pred = load_pred(rid)
    assert isinstance(pred, pd.Series)
    assert pred.index.nlevels == 2  # (datetime, instrument)
    assert len(pred) > 1000


def test_init_is_idempotent():
    init_qlib_once()
    init_qlib_once()  # second call should be a no-op, not raise


def test_init_raises_when_mlruns_missing(tmp_path, monkeypatch):
    """init_qlib_once must raise DependencyError(mlruns_missing) when mlruns dir doesn't exist."""
    from app.core import qlib_adapter
    from app.core.exceptions import DependencyError

    # reset module-level init flag so we can re-init
    qlib_adapter._initialized = False

    monkeypatch.setenv("QLIB_COMPANION_MLRUNS_DIR", str(tmp_path / "does_not_exist"))
    with pytest.raises(DependencyError) as excinfo:
        qlib_adapter.init_qlib_once()
    assert excinfo.value.code == "mlruns_missing"

    # restore for downstream tests
    qlib_adapter._initialized = False
    monkeypatch.undo()
    qlib_adapter.init_qlib_once()
