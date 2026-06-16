from app.models.recompute import phase_percent, fetch_metrics_chunked


def test_phase_percent_bounds():
    assert phase_percent("load", 1, 1) == 15
    assert phase_percent("score", 1, 1) == 30
    assert phase_percent("metrics", 0, 300) == 30
    assert phase_percent("metrics", 150, 300) == 60
    assert phase_percent("metrics", 300, 300) == 90
    assert phase_percent("enrich", 1, 1) == 100
    assert phase_percent("metrics", 0, 0) == 90  # total=0 -> frac=1.0 -> hi


def test_fetch_metrics_chunked_merges_and_reports():
    calls = []
    emits = []

    def fake_fetch(batch):
        calls.append(list(batch))
        return {s: {"v": s} for s in batch}

    syms = [f"S{i}" for i in range(125)]
    out = fetch_metrics_chunked(syms, fake_fetch, chunk_size=50,
                                emit=lambda done, total: emits.append((done, total)))
    assert len(out) == 125
    assert out["S124"] == {"v": "S124"}
    assert [len(c) for c in calls] == [50, 50, 25]
    assert emits == [(50, 125), (100, 125), (125, 125)]
