"""Smoke tests for the inference router."""
from fastapi.testclient import TestClient

from app.main import create_app


def _client():
    return TestClient(create_app())


def test_active_peek_returns_null_when_idle():
    resp = _client().get("/api/inference/active/peek")
    assert resp.status_code == 200
    assert resp.json() is None


def test_inference_status_returns_keys():
    resp = _client().get("/api/inference/status")
    assert resp.status_code == 200
    body = resp.json()
    assert "last_run_at" in body
    assert "last_success_at" in body
    assert "last_error" in body
    assert "is_running" in body


def test_get_job_404_when_missing():
    resp = _client().get("/api/inference/jobs/nonexistent")
    assert resp.status_code == 404


def test_internal_cache_invalidate_localhost_ok():
    """TestClient connects as 'testclient' which is in the allowed list."""
    resp = _client().post("/api/internal/cache/invalidate")
    assert resp.status_code == 200
    assert "cleared" in resp.json()
