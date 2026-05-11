from datetime import date
from pathlib import Path
from threading import Lock

import pandas as pd
import qlib
from qlib.constant import REG_CN, REG_US
from qlib.data import D
from qlib.workflow import R

from app.core.config import Settings
from app.core.exceptions import DependencyError, NotFoundError
from app.core.logging import get_logger

_log = get_logger("qlib_adapter")
_initialized = False
_lock = Lock()


def _resolve_mlruns_uri(settings: Settings) -> str:
    """Resolve mlruns_dir to an absolute file: URI.

    Settings.mlruns_dir defaults to ``examples/mlruns`` (repo-root relative).
    When the backend is run with cwd=backend/, we walk up one level to find it.
    Falls back to cwd-relative if the parent path doesn't exist.
    """
    candidate = Path(settings.mlruns_dir).expanduser()
    if not candidate.is_absolute():
        # Try cwd first
        cwd_candidate = (Path.cwd() / candidate).resolve()
        if cwd_candidate.is_dir():
            candidate = cwd_candidate
        else:
            # Try one level up (backend/ -> repo root)
            parent_candidate = (Path.cwd().parent / settings.mlruns_dir).resolve()
            if parent_candidate.is_dir():
                candidate = parent_candidate
            else:
                candidate = cwd_candidate
    else:
        candidate = candidate.resolve()
    return f"file:{candidate}"


def init_qlib_once(settings: Settings | None = None) -> None:
    """Idempotent qlib.init. Safe to call from many places."""
    global _initialized
    with _lock:
        if _initialized:
            return
        s = settings or Settings()
        region = REG_CN if s.qlib_region == "cn" else REG_US
        provider_uri = str(s.qlib_data_dir)
        if not Path(provider_uri).is_dir():
            raise DependencyError(
                f"qlib data not found at {provider_uri}",
                code="qlib_data_missing",
                context={"path": provider_uri},
            )
        mlruns_uri = _resolve_mlruns_uri(s)
        qlib.init(
            provider_uri=provider_uri,
            region=region,
            exp_manager={
                "class": "MLflowExpManager",
                "module_path": "qlib.workflow.expm",
                "kwargs": {
                    "uri": mlruns_uri,
                    "default_exp_name": "Experiment",
                },
            },
        )
        _initialized = True
        _log.info(
            "qlib_init_done",
            provider_uri=provider_uri,
            region=s.qlib_region,
            mlruns_uri=mlruns_uri,
        )


def get_ohlcv(symbols: list[str], start: str, end: str, freq: str = "day") -> pd.DataFrame:
    """Return MultiIndex DataFrame (datetime x instrument) with columns $open/$high/$low/$close/$volume/$factor."""
    init_qlib_once()
    fields = ["$open", "$high", "$low", "$close", "$volume", "$factor"]
    df = D.features(instruments=symbols, fields=fields, start_time=start, end_time=end, freq=freq)
    if df is None or df.empty:
        raise NotFoundError(
            f"no ohlcv for {symbols} between {start} and {end}",
            code="ohlcv_empty",
            context={"symbols": symbols, "start": start, "end": end},
        )
    return df


def get_calendar_end() -> date:
    init_qlib_once()
    cal = D.calendar(freq="day")
    if not len(cal):
        raise DependencyError("empty trading calendar", code="calendar_empty")
    return pd.Timestamp(cal[-1]).date()


def get_csi300_instruments() -> list[str]:
    init_qlib_once()
    inst_dict = D.instruments("csi300")
    inst_list = D.list_instruments(instruments=inst_dict, as_list=True)
    return sorted(inst_list)


def get_latest_recorder_id(experiment_name: str) -> str:
    init_qlib_once()
    try:
        exp = R.get_exp(experiment_name=experiment_name)
    except Exception as e:
        raise NotFoundError(
            f"experiment '{experiment_name}' not found",
            code="experiment_missing",
            context={"name": experiment_name},
        ) from e
    recs = exp.list_recorders()
    if not recs:
        raise NotFoundError(
            f"no recorders in experiment '{experiment_name}'",
            code="no_recorders",
            context={"experiment": experiment_name},
        )
    for rid in sorted(recs, key=lambda k: recs[k].info["start_time"], reverse=True):
        try:
            r = exp.get_recorder(recorder_id=rid)
            r.load_object("pred.pkl")
            return rid
        except Exception:
            continue
    raise NotFoundError(
        f"no recorder with pred.pkl in '{experiment_name}'",
        code="no_pred_pkl",
        context={"experiment": experiment_name},
    )


def load_pred(recorder_id: str, experiment_name: str = "daily_cn_fresh") -> pd.Series:
    init_qlib_once()
    exp = R.get_exp(experiment_name=experiment_name)
    rec = exp.get_recorder(recorder_id=recorder_id)
    pred = rec.load_object("pred.pkl")
    if isinstance(pred, pd.DataFrame):
        pred = pred["score"]
    return pred
