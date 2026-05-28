import functools
import logging
from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.config import Settings
from app.core.exceptions import NotFoundError
from app.core.qlib_adapter import (
    get_csi300_with_names,
    get_latest_close_prices,
    get_latest_recorder_id,
    init_qlib_once,
    load_pred,
)
from app.models.schemas import HorizonPrediction, ScreenItem

_log = logging.getLogger(__name__)

# Cached calibration map loaded from production/cache/latest_calibration.pkl
_CALIBRATION_CACHE: dict | None = None
_H_TO_N = {"1d": 1, "5d": 5, "20d": 20}


def _load_calibration() -> dict:
    global _CALIBRATION_CACHE
    if _CALIBRATION_CACHE is not None:
        return _CALIBRATION_CACHE
    try:
        # Avoid circular import; sys.path already includes repo root
        import sys
        repo_root = Path(__file__).resolve().parents[3]
        if str(repo_root) not in sys.path:
            sys.path.append(str(repo_root))
        from production.calibration import load_calibration
        _CALIBRATION_CACHE = load_calibration(
            repo_root / "production" / "cache" / "latest_calibration.pkl"
        )
    except Exception as exc:
        _log.warning("calibration_load_failed: %s", exc)
        _CALIBRATION_CACHE = {"maps": {}, "meta": {}}
    return _CALIBRATION_CACHE


def _get_qlib_latest_date() -> _date | None:
    try:
        from qlib.data import D
        cal = D.calendar()
        if len(cal) == 0:
            return None
        return pd.Timestamp(cal[-1]).date()
    except Exception as exc:
        _log.warning("qlib_calendar_load_failed: %s", exc)
        return None


def _next_n_trading_days(start: _date, n: int) -> _date:
    """Returns the date n trading days after `start`."""
    try:
        from qlib.data import D
        cal = D.calendar(start_time=str(start))
        if len(cal) > n:
            return pd.Timestamp(cal[n]).date()
        # Past the calendar — extrapolate via business days
        return (pd.Timestamp(cal[-1]) + pd.tseries.offsets.BDay(n - (len(cal) - 1))).date()
    except Exception:
        return (pd.Timestamp(start) + pd.tseries.offsets.BDay(n)).date()


def _name_map() -> dict[str, str]:
    """Build symbol -> name map from cn_names_cache.json directly.

    Covers all A-share stocks (~5500), not just CSI300, so the Picks page can
    show names for CSI500 + new listings that aren't yet in any market file.
    """
    import json
    from pathlib import Path
    cache_path = Path(__file__).resolve().parents[3] / "production" / "cn_names_cache.json"
    if not cache_path.exists():
        # Fall back to the old CSI300-only path if cache file is missing
        pairs = get_csi300_with_names()
        return {p["symbol"]: p["name"] for p in pairs}
    try:
        blob = json.loads(cache_path.read_text(encoding="utf-8"))
        bare_map = blob.get("map", {})
    except Exception:
        pairs = get_csi300_with_names()
        return {p["symbol"]: p["name"] for p in pairs}

    # Expand bare-code keys to fully-prefixed SH/SZ symbols by inferring exchange
    out: dict[str, str] = {}
    for bare, name in bare_map.items():
        if not isinstance(bare, str) or not isinstance(name, str) or not name:
            continue
        # Stocks: 6/9 prefix → SH (Shanghai), 0/2/3 → SZ (Shenzhen)
        # ETFs follow the same exchange convention.
        if bare[:1] in ("6", "9", "5"):
            sym = "SH" + bare
        elif bare[:1] in ("0", "1", "2", "3"):
            sym = "SZ" + bare
        elif bare[:1] in ("4", "8"):
            sym = "BJ" + bare
        else:
            continue
        out[sym] = name
    return out


def _build_screen_items(
    df: "pd.DataFrame",
    top: int,
    days: int,
    min_top: int,
    name_map: dict[str, str],
) -> list[ScreenItem]:
    import pandas as pd

    reserved = {"score", "consensus"}
    base_cols = [c for c in df.columns if c not in reserved]

    days_index = df.index.get_level_values("datetime").unique().sort_values()
    window = days_index[-days:]
    window_df = df.loc[df.index.get_level_values("datetime").isin(window)]

    window_df = window_df.assign(
        rank=window_df.groupby(level="datetime")["score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    last_day = window[-1]
    per_symbol = (
        window_df.groupby(level="instrument")
        .agg(
            score_avg=("score", "mean"),
            rank_avg=("rank", "mean"),
            days_in_top=("rank", lambda r: int((r <= top).sum())),
        )
        .sort_values("score_avg", ascending=False)
    )
    per_symbol = per_symbol[per_symbol["days_in_top"] >= min_top].head(top)

    last_slice = df.xs(last_day, level="datetime")

    items: list[ScreenItem] = []
    for rank_pos, (symbol, row) in enumerate(per_symbol.iterrows(), start=1):
        # score_today: look up from last_slice directly. If the symbol has no
        # data for last_day, fall back to the symbol's most recent score in the
        # window (rather than NaN, which breaks JSON serialization).
        if symbol in last_slice.index:
            score_today = float(last_slice.loc[symbol, "score"])
            consensus = (
                float(last_slice.loc[symbol, "consensus"])
                if "consensus" in last_slice.columns
                else 0.0
            )
            base_scores = {
                c: float(last_slice.loc[symbol, c])
                for c in base_cols
                if c in last_slice.columns and pd.notna(last_slice.loc[symbol, c])
            }
        else:
            # Symbol missing on last_day; fall back to the per-window average so
            # the response is still a valid JSON number.
            score_today = float(row["score_avg"])
            consensus = 0.0
            base_scores = {}

        items.append(
            ScreenItem(
                rank=rank_pos,
                symbol=symbol,
                name=name_map.get(symbol, ""),
                score_today=score_today,
                score_avg=float(row["score_avg"]),
                rank_avg=float(row["rank_avg"]),
                days_in_top=int(row["days_in_top"]),
                consensus=consensus,
                base_scores=base_scores,
            )
        )
    return items


def _passes_price(price: float | None, lo: float | None, hi: float | None) -> bool:
    """Inclusive price-range check; a None price fails any non-trivial bound."""
    if lo is None and hi is None:
        return True
    if price is None:
        return False
    if lo is not None and price < lo:
        return False
    if hi is not None and price > hi:
        return False
    return True


def candidates(
    top: int = 300,
    days: int = 5,
    min_top: int = 0,
    experiment: str | None = None,
    view: str = "ensemble",
    models: str | None = None,
) -> dict:
    """Return the full candidate pool (over-fetched, no filters) with complete
    Tier 1 metrics, cached per (recorder_id, view, models, top, days, min_top).

    Frontend pulls this ONCE per session/view and does filter + sort
    client-side. Cache invalidates automatically when a new recorder is trained
    (recorder_id is part of the key) or on backend restart.

    Args:
        models: comma-separated list of base column names to use for the
            ensemble score (e.g. "lgbm_1d,lgbm_5d,tra_5d"). When provided,
            the score is dynamically recomputed as -rank_avg over those
            columns only, overriding the pred.pkl's stored score. Unknown
            column names are silently dropped. None / empty / "all" → use
            the stored score (which was computed at pool time per the
            current ensemble convention, e.g. v9 = 1d+5d cols only).
            Takes precedence over `view`.
    """
    init_qlib_once()
    s = Settings()
    exp = experiment or s.default_experiment
    recorder_id = get_latest_recorder_id(exp)
    # Normalise models param into a hashable, order-independent key for caching.
    models_key: tuple[str, ...] = ()
    if models and models.strip().lower() not in ("", "all"):
        models_key = tuple(sorted({m.strip() for m in models.split(",") if m.strip()}))
    cached = _candidates_cached(recorder_id, exp, view, top, days, min_top, models_key)
    # Return a shallow copy so callers can mutate without poisoning the cache.
    return {**cached, "items": list(cached["items"])}


@functools.lru_cache(maxsize=32)
def _candidates_cached(
    recorder_id: str,
    exp: str,
    view: str,
    top: int,
    days: int,
    min_top: int,
    models_key: tuple[str, ...] = (),
) -> dict:
    """Heavy path: load pred.pkl, build screen items, fetch full metrics, populate
    all schema fields. Cached by lru_cache; do NOT call directly — use candidates()."""
    from app.core.qlib_adapter import get_filter_metrics
    from app.models.utils import is_st_name, parse_board

    pred = load_pred(recorder_id, experiment_name=exp)

    if isinstance(pred, pd.Series):
        df = pred.to_frame(name="score")
    else:
        df = pred.copy()
        if "score" not in df.columns:
            if df.shape[1] == 1:
                df = df.rename(columns={df.columns[0]: "score"})
            else:
                raise ValueError("pred frame missing 'score' column")

    if df.index.names != ["datetime", "instrument"]:
        df.index = df.index.set_names(["datetime", "instrument"])

    # Dynamic score recomputation: `models_key` takes precedence. If absent,
    # fall back to legacy `view` param. Empty key → use df["score"] as-is
    # from pred.pkl (already rank_avg + EWMA at pool time).
    score_cols: list[str] | None = None
    if models_key:
        score_cols = [c for c in models_key if c in df.columns]
    elif view != "ensemble":
        _view_prefix = {"lightgbm": "lgbm_", "alstm": "alstm_", "tra": "tra_"}
        prefix = _view_prefix.get(view, f"{view}_")
        score_cols = [c for c in df.columns if c.startswith(prefix)]

    if score_cols:
        df = df.copy()
        ranks = df[score_cols].groupby(level="datetime").rank(ascending=False, method="min")
        df["score"] = -ranks.mean(axis=1, skipna=True)
        # Re-apply EWMA so dynamic scores get the same temporal smoothing
        # the pool-time score has — adds ~30ms but keeps semantics consistent.
        from production.post_process import ewma_smooth
        df = ewma_smooth(df, alpha=0.5, score_col="score")

    dates = df.index.get_level_values("datetime").unique().sort_values()
    today = dates[-1]
    last_slice = df.xs(today, level="datetime")
    universe_size = int(last_slice["score"].count())

    name_map = _name_map()
    items = _build_screen_items(df, top=top, days=days, min_top=min_top, name_map=name_map)

    if items:
        prices = get_latest_close_prices([it.symbol for it in items])
        metrics = get_filter_metrics([it.symbol for it in items])
        for it in items:
            it.last_price = prices.get(it.symbol)
            m = metrics.get(it.symbol, {})
            it.pct_change_1d = m.get("pct_change_1d")
            it.pct_change_3d = m.get("pct_change_3d")
            it.pct_change_5d = m.get("pct_change_5d")
            it.pct_change_10d = m.get("pct_change_10d")
            it.pct_change_20d = m.get("pct_change_20d")
            it.amplitude = m.get("amplitude")
            it.vol_ratio = m.get("vol_ratio")
            it.is_new_high_20d = bool(m.get("is_new_high_20d", False))
            it.is_new_high_60d = bool(m.get("is_new_high_60d", False))
            it.is_new_high_120d = bool(m.get("is_new_high_120d", False))
            it.board = parse_board(it.symbol)
            it.is_st = is_st_name(it.name)

    available_models = sorted(
        c for c in df.columns
        if c not in ("score", "consensus")
        and not c.startswith("composite_")
        and not c.startswith("expected_")
    )

    # ===== Per-horizon enrichment + staleness ============================
    cal = _load_calibration()
    cal_maps = cal.get("maps", {})
    qlib_latest = _get_qlib_latest_date()
    latest_date = today.date() if hasattr(today, "date") else _date.fromisoformat(str(today)[:10])

    # Build per-horizon series at latest_date ONCE (not per item)
    horizon_data: dict[str, dict] = {}  # {horizon: {symbol: HorizonPrediction-shaped dict}}
    target_dates_map: dict[str, str] = {}
    try:
        last_slice = df.xs(today, level="datetime")
    except KeyError:
        last_slice = None

    if last_slice is not None and not last_slice.empty:
        for h in ("1d", "5d", "20d"):
            cols = [c for c in df.columns
                    if c.endswith(f"_{h}")
                    and not c.startswith("expected_")
                    and not c.startswith("composite_")
                    and c not in ("score", "consensus")]
            if not cols:
                continue
            # Composite score at latest_date
            sub = last_slice[cols]
            ranks = sub.rank(ascending=False, method="min")
            comp = -ranks.mean(axis=1, skipna=True)
            n = int(comp.notna().sum())
            if n == 0:
                continue
            comp_rank = comp.rank(ascending=False, method="min")
            target = _next_n_trading_days(latest_date, _H_TO_N[h])
            target_dates_map[h] = target.isoformat()

            iso = cal_maps.get(h)
            if iso is not None:
                try:
                    from production.calibration import apply_calibration
                    pr = apply_calibration(comp, iso)
                except Exception:
                    pr = pd.Series(index=comp.index, dtype=float)
            else:
                pr = pd.Series(np.nan, index=comp.index)

            # Build per-symbol dict
            h_map: dict[str, dict] = {}
            for sym in comp.index:
                sym_score = comp.loc[sym]
                if pd.isna(sym_score):
                    continue
                sym_rank = comp_rank.loc[sym]
                percentile = float(100.0 * (1.0 - (sym_rank - 1) / n)) if n > 0 else 0.0
                pred_return = pr.loc[sym]
                pred_return = None if pd.isna(pred_return) else float(pred_return)
                # raw_scores + agreement
                raw: dict[str, float] = {}
                for m in ("lgbm", "alstm", "tra"):
                    col = f"{m}_{h}"
                    if col in last_slice.columns:
                        v = last_slice.loc[sym, col] if sym in last_slice.index else np.nan
                        if pd.notna(v):
                            raw[m] = float(v)
                signs = [1 if v > 0 else (-1 if v < 0 else 0) for v in raw.values()]
                agreement = float(abs(sum(signs)) / len(signs)) if signs else None
                h_map[sym] = {
                    "target_date": target.isoformat(),
                    "pred_return": pred_return,
                    "percentile": percentile,
                    "model_agreement": agreement,
                    "raw_scores": raw,
                }
            horizon_data[h] = h_map

    # Attach horizons to each item (wrap each in HorizonPrediction so
    # ScreenItem.model_dump() round-trips cleanly)
    for it in items:
        hd: dict[str, HorizonPrediction] = {}
        for h, hmap in horizon_data.items():
            if it.symbol in hmap:
                hd[h] = HorizonPrediction(**hmap[it.symbol])
        if hd:
            it.horizons = hd

    # Staleness
    data_stale_days = 0
    data_latest_str = latest_date.isoformat()
    if qlib_latest and qlib_latest > latest_date:
        try:
            from qlib.data import D
            cal_seg = D.calendar(start_time=str(latest_date), end_time=str(qlib_latest))
            data_stale_days = max(0, len(cal_seg) - 1)
        except Exception:
            data_stale_days = (qlib_latest - latest_date).days
        data_latest_str = qlib_latest.isoformat()

    return {
        "experiment": exp,
        "recorder_id": recorder_id,
        "latest_date": data_latest_str,
        "as_of_date": latest_date.isoformat(),
        "data_latest_date": data_latest_str,
        "data_stale_days": data_stale_days,
        "window_days": days,
        "universe_size": universe_size,
        "available_models": available_models,
        "active_models": list(score_cols) if score_cols else None,
        "items": [it.model_dump() for it in items],
    }


def invalidate_candidates_cache() -> int:
    """Clear the lru_cache. Also drops the calibration cache so the next
    candidates() call picks up a freshly-written latest_calibration.pkl.
    Returns number of entries that were cleared.
    """
    global _CALIBRATION_CACHE
    _CALIBRATION_CACHE = None
    info = _candidates_cached.cache_info()
    _candidates_cached.cache_clear()
    return info.currsize


def screen(
    top: int = 30,
    days: int = 5,
    min_top: int = 0,
    experiment: str | None = None,
    view: str = "ensemble",
    min_price: float | None = None,
    max_price: float | None = None,
    pct_change_n: int = 5,
    min_pct_change: float | None = None,
    max_pct_change: float | None = None,
    min_amplitude: float | None = None,
    max_amplitude: float | None = None,
    min_vol_ratio: float | None = None,
    max_vol_ratio: float | None = None,
    new_high_n: int = 0,
    boards: str | None = None,
    exclude_st: bool = True,
) -> dict:
    """Rank + filter the model's universe.

    Filter pipeline (AND semantics, applied to over-fetched candidates):
      1. Existing: price range
      2. Tier 1: pct_change_{pct_change_n}d in [min,max]
      3. Tier 1: amplitude in [min,max]
      4. Tier 1: vol_ratio in [min,max]
      5. Tier 1: is_new_high_{new_high_n}d == True (if new_high_n != 0)
      6. Tier 1: board in boards (OR within multiselect)
      7. Tier 1: not is_st (if exclude_st)

    Symbols missing a metric fail any non-trivial bound on that metric.
    Returned items are re-ranked 1..N after filtering.
    """
    from app.core.qlib_adapter import get_filter_metrics
    from app.models.utils import Tier1FilterSpec, apply_tier1_filters, is_st_name, parse_board

    init_qlib_once()
    s = Settings()
    exp = experiment or s.default_experiment
    recorder_id = get_latest_recorder_id(exp)
    pred = load_pred(recorder_id, experiment_name=exp)

    if isinstance(pred, pd.Series):
        df = pred.to_frame(name="score")
    else:
        df = pred.copy()
        if "score" not in df.columns:
            if df.shape[1] == 1:
                df = df.rename(columns={df.columns[0]: "score"})
            else:
                raise ValueError("pred frame missing 'score' column")

    if df.index.names != ["datetime", "instrument"]:
        df.index = df.index.set_names(["datetime", "instrument"])

    if view != "ensemble":
        _view_prefix = {"lightgbm": "lgbm_", "alstm": "alstm_", "tra": "tra_"}
        prefix = _view_prefix.get(view, f"{view}_")
        view_cols = [c for c in df.columns if c.startswith(prefix)]
        if view_cols:
            df = df.copy()
            # rank-avg the matched cols (was simple mean, but rank-avg
            # matches pool-time semantics + handles scale differences)
            ranks = df[view_cols].groupby(level="datetime").rank(ascending=False, method="min")
            df["score"] = -ranks.mean(axis=1, skipna=True)

    dates = df.index.get_level_values("datetime").unique().sort_values()
    today = dates[-1]
    last_slice = df.xs(today, level="datetime")
    universe_size = int(last_slice["score"].count())

    name_map = _name_map()

    # Decide whether any expensive filter is active so we know whether to
    # over-fetch and run the metric pipeline.
    tier1_active = (
        min_price is not None or max_price is not None
        or min_pct_change is not None or max_pct_change is not None
        or min_amplitude is not None or max_amplitude is not None
        or min_vol_ratio is not None or max_vol_ratio is not None
        or new_high_n != 0
        or boards is not None
        or exclude_st
    )
    fetch_top = min(top * 4, 300) if tier1_active else top
    items = _build_screen_items(df, top=fetch_top, days=days, min_top=min_top, name_map=name_map)

    if items:
        prices = get_latest_close_prices([it.symbol for it in items])
        for it in items:
            it.last_price = prices.get(it.symbol)

        # Compute metrics + board + ST flags for every candidate
        metrics = get_filter_metrics([it.symbol for it in items])
        for it in items:
            m = metrics.get(it.symbol, {})
            it.pct_change_5d = m.get("pct_change_5d")
            it.amplitude = m.get("amplitude")
            it.vol_ratio = m.get("vol_ratio")
            it.board = parse_board(it.symbol)
            it.is_st = is_st_name(it.name)
            # Stash multi-N metrics in a transient dict so the filter pipeline
            # can read them without needing per-N schema fields.
            it.__dict__["_metrics"] = m

    # Build filter spec
    boards_set: set[str] | None = None
    if boards:
        boards_set = {b.strip() for b in boards.split(",") if b.strip()}

    spec = Tier1FilterSpec(
        pct_change_n=pct_change_n,
        min_pct_change=min_pct_change, max_pct_change=max_pct_change,
        min_amplitude=min_amplitude, max_amplitude=max_amplitude,
        min_vol_ratio=min_vol_ratio, max_vol_ratio=max_vol_ratio,
        new_high_n=new_high_n,
        boards=boards_set,
        exclude_st=exclude_st,
    )

    # Translate items -> dicts that apply_tier1_filters expects
    rows: list[dict] = []
    for it in items:
        m = it.__dict__.get("_metrics", {})
        rows.append({
            "symbol": it.symbol,
            "name": it.name,
            "board": it.board,
            "is_st": it.is_st,
            "amplitude": it.amplitude,
            "vol_ratio": it.vol_ratio,
            "pct_change_1d": m.get("pct_change_1d"),
            "pct_change_3d": m.get("pct_change_3d"),
            "pct_change_5d": it.pct_change_5d,
            "pct_change_10d": m.get("pct_change_10d"),
            "pct_change_20d": m.get("pct_change_20d"),
            "is_new_high_20d": m.get("is_new_high_20d", False),
            "is_new_high_60d": m.get("is_new_high_60d", False),
            "is_new_high_120d": m.get("is_new_high_120d", False),
            "_item": it,
        })

    # Existing price filter (kept in service, not Tier1FilterSpec)
    if min_price is not None or max_price is not None:
        rows = [r for r in rows if _passes_price(r["_item"].last_price, min_price, max_price)]

    # Tier 1 filters
    rows = apply_tier1_filters(rows, spec)

    # Re-rank and trim
    filtered_items = [r["_item"] for r in rows[:top]]
    for new_rank, it in enumerate(filtered_items, start=1):
        it.rank = new_rank

    # Strip transient metrics dict before serialization
    for it in filtered_items:
        it.__dict__.pop("_metrics", None)

    return {
        "experiment": exp,
        "recorder_id": recorder_id,
        "latest_date": str(today.date()),
        "window_days": days,
        "universe_size": universe_size,
        "items": [it.model_dump() for it in filtered_items],
    }


def prediction_history(
    symbol: str,
    days: int = 60,
    experiment: str | None = None,
    view: str = "ensemble",
) -> dict:
    """Return score + rank history for a single symbol.

    When `view` is not "ensemble", the unified `score` column is overridden
    with the row-wise mean of the per-model base columns matching that view's
    prefix (lightgbm -> lgbm_*, alstm -> alstm_*, tra -> tra_*). Each point's
    `base_scores` dict carries the per-base-column raw values for that day.
    """
    init_qlib_once()
    s = Settings()
    exp = experiment or s.default_experiment
    recorder_id = get_latest_recorder_id(exp)
    pred = load_pred(recorder_id, experiment_name=exp)

    # Normalize to a DataFrame with a `score` column. Keep extra columns
    # (consensus, base scores) if the new pred.pkl shape provides them.
    if isinstance(pred, pd.Series):
        df = pred.to_frame(name="score")
    else:
        df = pred.copy()
        if "score" not in df.columns:
            if df.shape[1] == 1:
                df = df.rename(columns={df.columns[0]: "score"})
            else:
                raise ValueError("pred frame missing 'score' column")

    # Normalize index names
    if df.index.names != ["datetime", "instrument"]:
        df.index = df.index.set_names(["datetime", "instrument"])

    if symbol not in df.index.get_level_values("instrument").unique():
        raise NotFoundError(
            f"no predictions for {symbol} in experiment {exp}",
            code="symbol_missing",
            context={"symbol": symbol, "experiment": exp},
        )

    # Override score for per-model views (same prefix map as screen())
    _view_prefix = {"lightgbm": "lgbm_", "alstm": "alstm_", "tra": "tra_"}
    if view in _view_prefix:
        prefix = _view_prefix[view]
        view_cols = [c for c in df.columns if c.startswith(prefix)]
        if view_cols:
            df = df.copy()
            df["score"] = df[view_cols].mean(axis=1)
        # else: silently fall through to ensemble score (old-shape pred.pkl)

    # Identify base columns for the per-point base_scores dict
    reserved = {"score", "consensus"}
    base_cols = [c for c in df.columns if c not in reserved]

    dates = df.index.get_level_values("datetime").unique().sort_values()
    window_dates = dates[-days:]
    window = df.loc[window_dates[0]:window_dates[-1]]

    # Per-day rank uses the (possibly overridden) score column
    daily_rank = (
        window["score"].groupby(level="datetime").rank(ascending=False, method="min")
    )
    universe_per_day = window["score"].groupby(level="datetime").count()

    # Slice to the requested symbol
    sym_window = window.xs(symbol, level="instrument").sort_index()
    sym_ranks = daily_rank.xs(symbol, level="instrument").sort_index()

    points = []
    for d, row in sym_window.iterrows():
        base_scores = {
            c: float(row[c])
            for c in base_cols
            if c in row.index and pd.notna(row[c])
        }
        points.append({
            "date": str(d.date()),
            "score": float(row["score"]),
            "rank": int(sym_ranks.loc[d]),
            "universe_size": int(universe_per_day.loc[d]),
            "base_scores": base_scores,
        })

    name_map = _name_map()
    return {
        "symbol": symbol,
        "name": name_map.get(symbol, ""),
        "experiment": exp,
        "points": points,
    }


def list_experiments() -> dict:
    """List mlflow experiments with their latest recorder + headline metrics."""
    init_qlib_once()
    from qlib.workflow import R

    out = []
    # qlib exposes the experiments dict via R.list_experiments(); fall back gracefully.
    try:
        exps = R.list_experiments()
    except Exception:
        exps = {}

    for name in exps:
        try:
            rid = get_latest_recorder_id(name)
            recorder = R.get_exp(experiment_name=name).get_recorder(recorder_id=rid)
            metrics: dict[str, float] = {}
            try:
                raw_metrics = recorder.list_metrics() or {}
                for k in ("IC", "ICIR", "Rank IC", "Rank ICIR"):
                    v = raw_metrics.get(k)
                    if v is not None:
                        metrics[k] = float(v)
            except Exception:
                pass
            out.append({
                "name": name,
                "latest_recorder_id": rid,
                "latest_metrics": metrics,
            })
        except Exception:
            continue

    return {"experiments": out}


def version_info() -> dict:
    """Return current/last/last-2 recorder metadata + next retrain ISO timestamp."""
    init_qlib_once()
    from qlib.workflow import R

    from app.core.config import Settings as _Settings
    settings = _Settings()
    recs = R.list_recorders(experiment_name=settings.retrain_recorder_experiment)
    sorted_recs = sorted(
        recs.values(), key=lambda rr: rr.info.get("start_time", ""), reverse=True
    )

    def _to_dto(rr) -> dict:
        metrics = {}
        if hasattr(rr, "list_metrics"):
            try:
                metrics = dict(rr.list_metrics().items())
            except Exception:
                metrics = {}
        return {
            "recorder_id": rr.id,
            "experiment": settings.retrain_recorder_experiment,
            "created_at": str(rr.info.get("start_time", "")),
            "metrics": metrics,
        }

    current = _to_dto(sorted_recs[0]) if len(sorted_recs) >= 1 else {
        "recorder_id": "", "experiment": settings.retrain_recorder_experiment,
        "created_at": "", "metrics": {},
    }
    previous = _to_dto(sorted_recs[1]) if len(sorted_recs) >= 2 else None
    previous_2 = _to_dto(sorted_recs[2]) if len(sorted_recs) >= 3 else None

    # Pull next retrain from the scheduler manager (T2/T4 exposes get_next_run_time)
    next_run = None
    try:
        from app.scheduling.router import get_manager as _get_scheduler

        mgr = _get_scheduler()
        nrt = mgr.get_next_run_time()
        next_run = nrt.isoformat() if nrt is not None else None
    except Exception:
        next_run = None

    return {
        "current": current,
        "previous": previous,
        "previous_2": previous_2,
        "next_retrain_at": next_run,
    }


def rollback_to(target: str = "previous_1") -> dict:
    """Move the current recorder's directory into production/archive/rolled_back/
    so the next /api/models/screen call picks the (formerly) previous recorder
    as the new current.

    target = "previous_1" archives 1 recorder.
    target = "previous_2" archives 2 recorders (rolls back two weeks).
    """
    import shutil
    from pathlib import Path
    from qlib.workflow import R

    init_qlib_once()
    from app.core.config import Settings as _Settings
    settings = _Settings()
    recs = sorted(
        R.list_recorders(experiment_name=settings.retrain_recorder_experiment).values(),
        key=lambda rr: rr.info.get("start_time", ""),
        reverse=True,
    )
    if len(recs) < 2:
        return {
            "status": "no_op",
            "archived_recorder_id": None,
            "new_current_recorder_id": None,
            "reason": "no_previous_recorder",
        }

    n_to_archive = 1 if target == "previous_1" else 2
    if len(recs) < n_to_archive + 1:
        return {
            "status": "no_op",
            "archived_recorder_id": None,
            "new_current_recorder_id": None,
            "reason": "insufficient_history",
        }

    mlruns_root = settings.mlruns_path
    archive_root = Path(__file__).resolve().parents[3] / "production" / "archive" / "rolled_back"

    archived_ids: list[str] = []
    for rec in recs[:n_to_archive]:
        rec_id = rec.id
        for exp_dir in mlruns_root.iterdir():
            if not exp_dir.is_dir():
                continue
            src = exp_dir / rec_id
            if src.is_dir():
                dest = archive_root / exp_dir.name / rec_id
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dest))
                archived_ids.append(rec_id)
                break

    new_current = recs[n_to_archive].id if len(recs) > n_to_archive else None
    return {
        "status": "rolled_back" if archived_ids else "no_op",
        "archived_recorder_id": ",".join(archived_ids) if archived_ids else None,
        "new_current_recorder_id": new_current,
        "reason": None if archived_ids else "recorder_dir_not_found",
    }
