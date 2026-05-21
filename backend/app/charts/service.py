import pandas as pd

from app.charts.schemas import CandleBar, ChartPayload, PredictionBar
from app.core.config import Settings
from app.core.exceptions import NotFoundError
from app.core.qlib_adapter import (
    get_calendar_end,
    get_latest_recorder_id,
    get_ohlcv,
    init_qlib_once,
    load_pred,
    next_trading_days,
)


def get_chart(
    symbol: str,
    start: str,
    end: str,
    with_pred: bool = True,
    experiment: str | None = None,
) -> ChartPayload:
    """Build a chart payload with actual + (optionally) predicted + forecast bars.

    Time alignment:
      - For each actual bar at trading day D[i] (i >= 2), the predicted bar at D[i]
        is computed from score[D[i-2]]: predicted_open = close[D[i-1]],
        predicted_close = close[D[i-1]] * (1 + score[D[i-2]]).
      - Forecast bars extend 1-2 trading days past last actual using the same logic.
    """
    init_qlib_once()
    settings = Settings()
    experiment = experiment or settings.default_experiment

    try:
        df = get_ohlcv([symbol], start=start, end=end)
    except NotFoundError as e:
        raise NotFoundError(
            f"no data for {symbol} between {start} and {end}",
            code="ohlcv_empty",
            context={"symbol": symbol},
        ) from e

    if df.empty:
        raise NotFoundError(
            f"no data for {symbol} between {start} and {end}",
            code="ohlcv_empty",
            context={"symbol": symbol},
        )

    # df has MultiIndex (instrument, datetime); pull the symbol slice
    if symbol not in df.index.get_level_values("instrument").unique():
        raise NotFoundError(
            f"symbol {symbol} not in dataset",
            code="symbol_missing",
            context={"symbol": symbol},
        )
    sub = df.xs(symbol, level="instrument").sort_index()
    actual: list[CandleBar] = [
        CandleBar(
            time=str(idx.date()),
            open=float(row["$open"]),
            high=float(row["$high"]),
            low=float(row["$low"]),
            close=float(row["$close"]),
            volume=float(row["$volume"]),
        )
        for idx, row in sub.iterrows()
    ]

    predicted: list[PredictionBar] = []
    forecast: list[PredictionBar] = []
    meta: dict = {
        "last_actual_date": actual[-1].time if actual else None,
        "with_pred": with_pred,
        "experiment": experiment,
    }

    if with_pred and len(actual) >= 3:
        recorder_id = get_latest_recorder_id(experiment)
        meta["recorder_id"] = recorder_id
        pred_series = load_pred(recorder_id, experiment_name=experiment)
        if symbol in pred_series.index.get_level_values("instrument").unique():
            scores = pred_series.xs(symbol, level="instrument").sort_index()
            score_map = {str(t.date()): float(v) for t, v in scores.items()}

            actual_dates = [b.time for b in actual]
            for i in range(2, len(actual)):
                sig_date = actual_dates[i - 2]
                if sig_date not in score_map:
                    continue
                sc = score_map[sig_date]
                prev_close = actual[i - 1].close
                pred_close = prev_close * (1 + sc)
                open_, close_ = prev_close, pred_close
                spread = max(0.001 * prev_close, abs(close_ - open_) * 0.08)
                predicted.append(
                    PredictionBar(
                        time=actual[i].time,
                        open=open_,
                        high=max(open_, close_) + spread,
                        low=min(open_, close_) - spread,
                        close=close_,
                        score=sc,
                    )
                )

            # Forecast: future bars beyond last actual.
            # score[last] predicts target = last + 2 trading days.
            # score[last-1] predicts target = last + 1 trading day.
            calendar_end = get_calendar_end()
            last_actual_date = pd.Timestamp(actual[-1].time)
            # naive future calendar using qlib's trading calendar for next 5 business days
            future_cal = next_trading_days(last_actual_date, n=2)
            if len(future_cal) >= 1 and len(actual) >= 2:
                sig_date_for_f1 = actual_dates[-2]
                if sig_date_for_f1 in score_map:
                    sc1 = score_map[sig_date_for_f1]
                    prev_close = actual[-1].close
                    close_f1 = prev_close * (1 + sc1)
                    spread = max(0.001 * prev_close, abs(close_f1 - prev_close) * 0.08)
                    forecast.append(
                        PredictionBar(
                            time=str(future_cal[0].date()),
                            open=prev_close,
                            high=max(prev_close, close_f1) + spread,
                            low=min(prev_close, close_f1) - spread,
                            close=close_f1,
                            score=sc1,
                        )
                    )
                    if len(future_cal) >= 2:
                        sig_date_for_f2 = actual_dates[-1]
                        if sig_date_for_f2 in score_map:
                            sc2 = score_map[sig_date_for_f2]
                            close_f2 = close_f1 * (1 + sc2)
                            spread2 = max(0.001 * close_f1, abs(close_f2 - close_f1) * 0.08)
                            forecast.append(
                                PredictionBar(
                                    time=str(future_cal[1].date()),
                                    open=close_f1,
                                    high=max(close_f1, close_f2) + spread2,
                                    low=min(close_f1, close_f2) - spread2,
                                    close=close_f2,
                                    score=sc2,
                                )
                            )
            meta["calendar_end"] = str(calendar_end)

    return ChartPayload(symbol=symbol, actual=actual, predicted=predicted, forecast=forecast, meta=meta)
