from app.models.schemas import ScreenItem
from app.analysis.schemas import AiAnalysis


def test_screenitem_has_ai_analysis_field():
    it = ScreenItem(rank=1, symbol="SH600519", score_today=0.9, score_avg=0.9,
                    rank_avg=1.0, days_in_top=1)
    assert it.ai_analysis is None
    it2 = it.model_copy(update={"ai_analysis": AiAnalysis(interpretation="x", stance="neutral")})
    assert it2.ai_analysis.interpretation == "x"
    assert it.ai_analysis is None  # original untouched


def test_candidates_route_attaches_analysis(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.models import router as models_router
    from app.core.db import get_session

    monkeypatch.setattr(models_router.service, "candidates", lambda **kw: {
        "experiment": "e", "recorder_id": "r", "latest_date": "2026-06-10",
        "window_days": 5, "universe_size": 1, "available_models": [],
        "items": [ScreenItem(rank=1, symbol="SH600519", score_today=1.0,
                             score_avg=1.0, rank_avg=1.0, days_in_top=1)],
        "as_of_date": "2026-06-10",
    })

    async def fake_fetch(session, symbols, as_of):
        return {"SH600519": AiAnalysis(interpretation="超跌反弹", stance="favorable")}
    monkeypatch.setattr(models_router, "fetch_analyses", fake_fetch)

    app = FastAPI()
    app.include_router(models_router.router, prefix="/api/models")

    async def _noop_session():
        yield None
    app.dependency_overrides[get_session] = _noop_session

    client = TestClient(app)
    r = client.get("/api/models/candidates")
    body = r.json()
    assert body["items"][0]["ai_analysis"]["interpretation"] == "超跌反弹"
