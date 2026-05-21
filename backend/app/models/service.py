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


def screen(
    top: int = 30,
    days: int = 5,
    min_top: int = 0,
    experiment: str | None = None,
    view: str = "ensemble",
) -> dict:
    """
    Rank the model's universe by 'score_avg over last `days` days', then filter by
    'days_in_top >= min_top' if specified. Returns at most `top` items.

    When `view` is not "ensemble", the unified `score` column is overridden with
    the row-wise mean of the per-model base columns matching that view's prefix
    (lightgbm -> lgbm_, alstm -> alstm_, tra -> tra_). Falls back to the
    ensemble score if no matching base columns are present in the prediction
    frame (e.g. old-shape pred.pkl).
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

    # If a per-model view is requested, override `score` with the row-wise mean
    # of the matching base columns. Map UI-friendly view names to the actual
    # column prefix used in pred.pkl (lightgbm -> lgbm_*, etc.).
    if view != "ensemble":
        _view_prefix = {
            "lightgbm": "lgbm_",
            "alstm": "alstm_",
            "tra": "tra_",
        }
        prefix = _view_prefix.get(view, f"{view}_")
        view_cols = [c for c in df.columns if c.startswith(prefix)]
        if view_cols:
            df = df.copy()
            df["score"] = df[view_cols].mean(axis=1)
        # else: silently fall through to existing ensemble score (no per-model
        # cols available — e.g. old-shape pred.pkl).

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
