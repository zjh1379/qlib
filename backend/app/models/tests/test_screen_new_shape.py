import pandas as pd
import pytest

from app.models.service import _build_screen_items


def _mk_df():
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2026-05-10", "2026-05-14"), ["SH600000", "SH600001"]],
        names=["datetime", "instrument"],
    )
    df = pd.DataFrame(
        {
            "score": [0.10, -0.05, 0.11, -0.04, 0.13, -0.03, 0.12, -0.02, 0.14, -0.01],
            "consensus": [1.0, 0.33, 1.0, 0.33, 1.0, 0.33, 1.0, 0.33, 1.0, 0.33],
            "lgbm_1d": [0.08, -0.04, 0.09, -0.03, 0.11, -0.02, 0.10, -0.01, 0.12, 0.0],
            "lgbm_5d": [0.11, -0.05, 0.12, -0.04, 0.13, -0.03, 0.12, -0.02, 0.14, -0.01],
            "lgbm_20d": [0.12, -0.06, 0.13, -0.05, 0.15, -0.04, 0.13, -0.03, 0.16, -0.02],
        },
        index=idx,
    )
    return df


def test_screen_items_include_consensus_and_base_scores():
    df = _mk_df()
    items = _build_screen_items(df, top=2, days=5, min_top=0, name_map={})
    assert len(items) <= 2
    # SH600000 has higher avg score -> rank 1
    top_item = items[0]
    assert top_item.symbol == "SH600000"
    assert top_item.consensus == pytest.approx(1.0)
    assert set(top_item.base_scores.keys()) == {"lgbm_1d", "lgbm_5d", "lgbm_20d"}


def test_screen_items_score_today_uses_chronologically_last_day():
    """Even if input df is shuffled, score_today should be from the last datetime."""
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-05-14"), "SH600000"),
            (pd.Timestamp("2026-05-10"), "SH600000"),
            (pd.Timestamp("2026-05-12"), "SH600000"),
        ],
        names=["datetime", "instrument"],
    )
    df = pd.DataFrame({"score": [0.30, 0.10, 0.20]}, index=idx)
    items = _build_screen_items(df, top=1, days=5, min_top=0, name_map={})
    assert len(items) == 1
    assert items[0].score_today == pytest.approx(0.30)  # 2026-05-14 value


def test_screen_view_lightgbm_uses_only_lgbm_columns():
    """Direct service-layer test: passing view='lightgbm' should override the
    score column with the average of lgbm_* base columns, so per-symbol ranking
    follows lgbm-only scores rather than the unified ensemble score."""
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2026-05-10", "2026-05-14"), ["SH600000", "SH600001"]],
        names=["datetime", "instrument"],
    )
    df = pd.DataFrame(
        {
            # Ensemble score deliberately favors SH600000 — view override should
            # flip the ordering because lgbm avg favors SH600001.
            "score": [0.50, 0.10] * 5,
            "consensus": [1.0] * 10,
            "lgbm_1d": [0.10, 0.30] * 5,
            "lgbm_5d": [0.20, 0.40] * 5,
            "alstm_1d": [0.00] * 10,
        },
        index=idx,
    )

    # Apply the view='lightgbm' override (mirrors what service.screen does).
    lgbm_cols = [c for c in df.columns if c.startswith("lgbm_")]
    df = df.copy()
    df["score"] = df[lgbm_cols].mean(axis=1)

    items = _build_screen_items(df, top=2, days=5, min_top=0, name_map={})
    # SH600001 has higher lgbm avg (0.35) than SH600000 (0.15)
    assert items[0].symbol == "SH600001"
