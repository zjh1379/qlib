import pytest

from app.evaluation.service import list_recorders_with_summary
from app.core.qlib_adapter import init_qlib_once


@pytest.fixture(scope="module")
def qlib_ready():
    try:
        init_qlib_once()
    except Exception as exc:
        pytest.skip(f"qlib not initializable: {exc}")


def test_list_recorders_returns_at_least_one_known(qlib_ready):
    summaries = list_recorders_with_summary()
    # The dev environment must have at least the daily_cn_fresh recorder
    assert any(s.experiment == "daily_cn_fresh" for s in summaries), \
        "expected daily_cn_fresh experiment to have at least one recorder"


def test_summary_fields_are_populated(qlib_ready):
    summaries = list_recorders_with_summary()
    if not summaries:
        pytest.skip("no recorders available")
    s = summaries[0]
    assert s.recorder_id
    assert s.experiment
    assert s.run_name
    assert s.created_at
    # pred_start/end/rows may be None on errors but should usually be set
    if s.pred_rows is not None:
        assert s.pred_rows > 0
    assert s.has_eval is False  # cache empty on first call
