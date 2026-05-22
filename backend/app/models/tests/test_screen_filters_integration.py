"""End-to-end tests for the screener filter pipeline via the FastAPI test client.

These hit the real qlib data store and the real default mlruns recorder; they
skip if either is unavailable.
"""
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app


@pytest.fixture
async def client():
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        async with app.router.lifespan_context(app):
            yield c


async def _try_screen(client, **params) -> dict:
    qs = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
    r = await client.get(f"/api/models/screen?{qs}")
    return r.status_code, r.json()


async def test_baseline_returns_items_with_new_fields(client):
    status, body = await _try_screen(client, top=3, exclude_st="false")
    if status == 404:
        pytest.skip("no recorder in default experiment")
    assert status == 200
    assert len(body["items"]) <= 3
    for it in body["items"]:
        # New Tier 1 fields are present (may be None when qlib data missing)
        for k in ("board", "is_st", "amplitude", "vol_ratio", "pct_change_5d"):
            assert k in it


async def test_board_filter_main_only(client):
    status, body = await _try_screen(client, top=20, boards="main", exclude_st="false")
    if status == 404:
        pytest.skip("no recorder")
    assert status == 200
    for it in body["items"]:
        assert it["board"] == "main"


async def test_exclude_st_default_drops_st(client):
    status, body = await _try_screen(client, top=30, exclude_st="true")
    if status == 404:
        pytest.skip("no recorder")
    assert status == 200
    for it in body["items"]:
        assert it["is_st"] is False


async def test_bad_pct_change_n_returns_400(client):
    status, body = await _try_screen(client, top=5, pct_change_n=7)
    assert status == 400
    assert body["code"] == "bad_pct_change_n"


async def test_bad_new_high_n_returns_400(client):
    status, body = await _try_screen(client, top=5, new_high_n=30)
    assert status == 400
    assert body["code"] == "bad_new_high_n"


async def test_filtered_items_renumbered_from_1(client):
    status, body = await _try_screen(client, top=10, boards="main", exclude_st="true")
    if status == 404:
        pytest.skip("no recorder")
    assert status == 200
    items = body["items"]
    if items:
        # Ranks must be 1..N contiguous after filtering
        ranks = [it["rank"] for it in items]
        assert ranks == list(range(1, len(items) + 1))
