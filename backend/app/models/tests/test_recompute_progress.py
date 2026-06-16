from app.models.recompute import phase_percent


def test_phase_percent_bounds():
    assert phase_percent("load", 1, 1) == 10
    assert phase_percent("score", 1, 1) == 15
    assert phase_percent("metrics", 0, 10) == 15
    assert phase_percent("metrics", 1, 10) == 23   # 15 + (92-15)*0.1
    assert phase_percent("metrics", 10, 10) == 92
    assert phase_percent("enrich", 1, 1) == 100
    assert phase_percent("metrics", 0, 0) == 92  # total=0 -> frac=1.0 -> hi
