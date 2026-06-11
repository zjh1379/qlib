import production.daily_inference as di


def test_post_analysis_refresh_posts_to_internal_url(monkeypatch):
    calls = {}

    class _Resp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout=0):
        calls["url"] = req.full_url
        calls["method"] = req.get_method()
        return _Resp()

    monkeypatch.setattr(di.urllib.request, "urlopen", fake_urlopen)
    di._post_analysis_refresh()
    assert calls["url"] == "http://127.0.0.1:8000/api/internal/analysis/refresh"
    assert calls["method"] == "POST"


def test_post_analysis_refresh_failsoft(monkeypatch):
    def boom(req, timeout=0):
        raise OSError("backend down")
    monkeypatch.setattr(di.urllib.request, "urlopen", boom)
    di._post_analysis_refresh()  # must not raise
