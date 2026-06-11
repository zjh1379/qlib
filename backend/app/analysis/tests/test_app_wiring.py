def test_analysis_routes_registered():
    from app.main import create_app
    app = create_app()
    paths = {r.path for r in app.routes}
    assert "/api/analysis/status" in paths
    assert "/api/internal/analysis/refresh" in paths
