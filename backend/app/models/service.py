import pandas as pd

from app.core.config import Settings
from app.core.exceptions import NotFoundError
from app.core.qlib_adapter import (
    get_csi300_with_names,
    get_latest_recorder_id,
    init_qlib_once,
    load_pred,
)
from app.models.schemas import ScreenItem


def _name_map() -> dict[str, str]:
    """Build symbol -> name map from qlib_adapter.get_csi300_with_names()."""
    pairs = get_csi300_with_names()
    return {p["symbol"]: p["name"] for p in pairs}


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


def screen(top: int = 30, days: int = 5, min_top: int = 0, experiment: str | None = None) -> dict:
    """
    Rank the model's universe by 'score_avg over last `days` days', then filter by
    'days_in_top >= min_top' if specified. Returns at most `top` items.
    """
    init_qlib_once()
    s = Settings()
    exp = experiment or s.default_experiment
    recorder_id = get_latest_recorder_id(exp)
    pred = load_pred(recorder_id, experiment_name=exp)

    # Normalize to DataFrame with a `score` column. Keep extra columns
    # (consensus, base scores) if the new pred.pkl shape provides them.
    if isinstance(pred, pd.Series):
        df = pred.to_frame(name="score")
    else:
        df = pred.copy()
        if "score" not in df.columns:
            # Defensive: if the prediction frame has no `score` column but has a
            # single column, treat that as score. Otherwise this is malformed.
            if df.shape[1] == 1:
                df = df.rename(columns={df.columns[0]: "score"})
            else:
                raise ValueError("pred frame missing 'score' column")

    # Ensure the index names match what _build_screen_items expects.
    if df.index.names != ["datetime", "instrument"]:
        df.index = df.index.set_names(["datetime", "instrument"])

    # Compute universe_size and last_day from the prediction frame.
    dates = df.index.get_level_values("datetime").unique().sort_values()
    today = dates[-1]
    last_slice = df.xs(today, level="datetime")
    universe_size = int(last_slice["score"].count())

    name_map = _name_map()

    items = _build_screen_items(df, top=top, days=days, min_top=min_top, name_map=name_map)

    return {
        "experiment": exp,
        "recorder_id": recorder_id,
        "latest_date": str(today.date()),
        "window_days": days,
        "universe_size": universe_size,
        "items": [it.model_dump() for it in items],
    }


def prediction_history(symbol: str, days: int = 60, experiment: str | None = None) -> dict:
    """Return score + rank history for a single symbol."""
    init_qlib_once()
    s = Settings()
    exp = experiment or s.default_experiment
    recorder_id = get_latest_recorder_id(exp)
    pred = load_pred(recorder_id, experiment_name=exp)
    if isinstance(pred, pd.DataFrame):
        pred = pred["score"]

    if symbol not in pred.index.get_level_values(1).unique():
        raise NotFoundError(
            f"no predictions for {symbol} in experiment {exp}",
            code="symbol_missing",
            context={"symbol": symbol, "experiment": exp},
        )

    dates = pred.index.get_level_values(0).unique().sort_values()
    window_dates = dates[-days:]
    window = pred.loc[window_dates[0]:window_dates[-1]]
    daily_rank = window.groupby(level=0).rank(ascending=False, method="min")
    sym_scores = window.xs(symbol, level=1).sort_index()
    sym_ranks = daily_rank.xs(symbol, level=1).sort_index()
    universe_per_day = window.groupby(level=0).count()

    points = []
    for d in sym_scores.index:
        points.append({
            "date": str(d.date()),
            "score": float(sym_scores.loc[d]),
            "rank": int(sym_ranks.loc[d]),
            "universe_size": int(universe_per_day.loc[d]),
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
