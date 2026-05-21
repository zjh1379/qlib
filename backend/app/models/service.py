import pandas as pd

from app.core.config import Settings
from app.core.exceptions import NotFoundError
from app.core.qlib_adapter import (
    get_csi300_with_names,
    get_latest_recorder_id,
    init_qlib_once,
    load_pred,
)


def _name_map() -> dict[str, str]:
    """Build symbol -> name map from qlib_adapter.get_csi300_with_names()."""
    pairs = get_csi300_with_names()
    return {p["symbol"]: p["name"] for p in pairs}


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
    if isinstance(pred, pd.DataFrame):
        pred = pred["score"]

    # Window = last `days` trading days available in pred
    dates = pred.index.get_level_values(0).unique().sort_values()
    window_dates = dates[-days:]
    today = window_dates[-1]
    window = pred.loc[window_dates[0]:today]

    # Cross-sectional rank per day (lower = better)
    daily_rank = window.groupby(level=0).rank(ascending=False, method="min")

    # Compute aggregates per symbol
    score_today = pred.loc[today]
    score_avg = window.groupby(level=1).mean()
    rank_avg = daily_rank.groupby(level=1).mean()
    days_in_top = (daily_rank <= top).groupby(level=1).sum()

    universe_size = int(pred.loc[today].count())
    name_map = _name_map()

    # Build sorted list
    df = pd.DataFrame({
        "score_today": score_today,
        "score_avg": score_avg,
        "rank_avg": rank_avg,
        "days_in_top": days_in_top,
    }).dropna(subset=["score_today"])

    if min_top > 0:
        df = df[df["days_in_top"] >= min_top]

    df = df.sort_values("score_avg", ascending=False).head(top)

    items = []
    for i, (symbol, row) in enumerate(df.iterrows(), start=1):
        items.append({
            "rank": i,
            "symbol": symbol,
            "name": name_map.get(symbol, ""),
            "score_today": float(row["score_today"]),
            "score_avg": float(row["score_avg"]),
            "rank_avg": float(row["rank_avg"]),
            "days_in_top": int(row["days_in_top"]),
        })

    return {
        "experiment": exp,
        "recorder_id": recorder_id,
        "latest_date": str(today.date()),
        "window_days": days,
        "universe_size": universe_size,
        "items": items,
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
