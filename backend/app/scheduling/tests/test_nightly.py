from unittest.mock import MagicMock
from app.scheduling.service import SchedulerManager


def test_install_nightly_inference_adds_cron_job():
    mgr = SchedulerManager(job_fn=MagicMock())
    mgr._scheduler = MagicMock()
    mgr.install_nightly_inference(enabled=True, hour=2)
    assert mgr._scheduler.add_job.called
    kwargs = mgr._scheduler.add_job.call_args.kwargs
    assert kwargs.get("id") == "nightly_inference"


def test_install_nightly_inference_disabled_is_noop():
    mgr = SchedulerManager(job_fn=MagicMock())
    mgr._scheduler = MagicMock()
    mgr.install_nightly_inference(enabled=False, hour=2)
    mgr._scheduler.add_job.assert_not_called()
