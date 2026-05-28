"""Verify data refresh success triggers inference job."""
from unittest import mock

from app.data import service as data_service


def test_refresh_success_triggers_inference():
    with mock.patch("app.inference.service.trigger_inference") as mock_trigger:
        mock_trigger.return_value = mock.Mock(status="started", job_id="abc")
        data_service._on_refresh_success("job-xyz")
        mock_trigger.assert_called_once()
        kwargs = mock_trigger.call_args.kwargs
        assert kwargs.get("reason") == "data_refresh"


def test_refresh_callback_swallows_exception():
    """If trigger raises, we shouldn't crash the refresh thread."""
    with mock.patch("app.inference.service.trigger_inference",
                    side_effect=RuntimeError("nope")):
        # Should not raise
        data_service._on_refresh_success("job-xyz")
