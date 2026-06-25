from app.core.config import Settings


def test_nightly_defaults_off():
    s = Settings()
    assert s.nightly_inference_enabled is False
    assert 0 <= s.nightly_inference_hour <= 23
