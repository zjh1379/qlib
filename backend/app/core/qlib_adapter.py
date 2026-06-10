from datetime import date, timedelta
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
    """Return a file:// URI for the mlruns directory. Raises DependencyError if missing."""
    path = settings.mlruns_path
    if not path.is_dir():
        raise DependencyError(
            f"mlruns directory not found at {path}",
            code="mlruns_missing",
            context={"path": str(path)},
        )
    return f"file:{path}"


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


def get_latest_close_prices(symbols: list[str], lookback_days: int = 10) -> dict[str, float]:
    """Return {symbol: most-recent-non-null close price} for each requested symbol.

    `lookback_days` controls how far back we search per symbol — a value of 10 covers
    typical suspensions/holiday gaps without pulling unnecessary history. Symbols with
    no data in the window are omitted from the returned dict.
    """
    init_qlib_once()
    if not symbols:
        return {}
    end_date = get_calendar_end()
    start_date = end_date - timedelta(days=lookback_days)
    try:
        df = D.features(
            instruments=symbols,
            fields=["$close"],
            start_time=start_date.isoformat(),
            end_time=end_date.isoformat(),
        )
    except Exception as exc:
        _log.warning("latest_close_fetch_failed error=%s", str(exc))
        return {}
    if df is None or df.empty:
        return {}
    # qlib returns MultiIndex (instrument, datetime). For each instrument, take the
    # last non-null close. groupby().last() ignores NaN if dropna() is called first.
    df = df.dropna()
    if df.empty:
        return {}
    out: dict[str, float] = {}
    for inst, group in df.groupby(level="instrument"):
        out[inst] = float(group["$close"].iloc[-1])
    return out


def get_filter_metrics(
    symbols: list[str],
    end_date: date | None = None,
    lookback_days: int = 200,
) -> dict[str, dict]:
    """Batch-compute Tier 1 filter metrics for a set of symbols in one qlib call.

    Returned dict per symbol:
        {
          "last_close": float,
          "pct_change_1d": float,    # (close_T / close_T-1) - 1
          "pct_change_3d": float,
          "pct_change_5d": float,
          "pct_change_10d": float,
          "pct_change_20d": float,
          "amplitude": float,        # (high_T - low_T) / close_T-1
          "vol_ratio": float,        # vol_T / mean(vol_T-1..T-5)
          "is_new_high_20d": bool,   # close_T == max(close, T-19..T)
          "is_new_high_60d": bool,
          "is_new_high_120d": bool,
        }

    Symbols with insufficient history (e.g. listed within `lookback_days`) are
    returned with NaN-safe defaults: the relevant pct_change_N becomes 0 and
    new-high flags become False. last_close is always populated when the symbol
    has any OHLCV in the window; otherwise the symbol is omitted from the result.
    """
    init_qlib_once()
    if not symbols:
        return {}
    if end_date is None:
        end_date = get_calendar_end()
    start_date = end_date - timedelta(days=lookback_days)

    try:
        df = D.features(
            instruments=symbols,
            fields=["$open", "$high", "$low", "$close", "$volume"],
            start_time=start_date.isoformat(),
            end_time=end_date.isoformat(),
        )
    except Exception as exc:
        _log.warning("filter_metrics_fetch_failed error=%s", str(exc))
        return {}
    if df is None or df.empty:
        return {}

    out: dict[str, dict] = {}
    for inst, group in df.groupby(level="instrument"):
        g = group.dropna(subset=["$close"])
        if g.empty:
            continue
        closes = g["$close"].to_numpy()
        highs = g["$high"].to_numpy()
        lows = g["$low"].to_numpy()
        vols = g["$volume"].to_numpy()
        last = closes[-1]

        def _pct_n(n: int) -> float:
            if len(closes) <= n:
                return 0.0
            prev = closes[-1 - n]
            return float((last / prev) - 1.0) if prev > 0 else 0.0

        amp = 0.0
        if len(closes) >= 2 and closes[-2] > 0:
            amp = float((highs[-1] - lows[-1]) / closes[-2])

        vol_ratio = 0.0
        if len(vols) >= 6:
            past5_mean = vols[-6:-1].mean()
            if past5_mean > 0:
                vol_ratio = float(vols[-1] / past5_mean)

        def _is_new_high(n: int) -> bool:
            if len(closes) < n:
                return False
            window = closes[-n:]
            # Use a tiny epsilon to absorb float noise
            return bool(last + 1e-9 >= window.max())

        out[inst] = {
            "last_close": float(last),
            "pct_change_1d": _pct_n(1),
            "pct_change_3d": _pct_n(3),
            "pct_change_5d": _pct_n(5),
            "pct_change_10d": _pct_n(10),
            "pct_change_20d": _pct_n(20),
            "amplitude": amp,
            "vol_ratio": vol_ratio,
            "is_new_high_20d": _is_new_high(20),
            "is_new_high_60d": _is_new_high(60),
            "is_new_high_120d": _is_new_high(120),
        }
    return out


def get_calendar_end() -> date:
    init_qlib_once()
    cal = D.calendar(freq="day")
    if not len(cal):
        raise DependencyError("empty trading calendar", code="calendar_empty")
    return pd.Timestamp(cal[-1]).date()


def next_trading_days(after: str | pd.Timestamp, n: int = 2) -> list[pd.Timestamp]:
    """Return up to `n` trading days strictly after `after` (a date string or Timestamp)."""
    init_qlib_once()
    anchor = pd.Timestamp(after)
    cal = D.calendar(start_time=anchor, end_time=anchor + pd.Timedelta(days=10 + n * 3))
    return [pd.Timestamp(d) for d in cal if pd.Timestamp(d) > anchor][:n]


def get_csi300_instruments() -> list[str]:
    """Compatibility wrapper. Prefer get_instruments_for_market('csi300')."""
    init_qlib_once()
    inst_dict = D.instruments("csi300")
    inst_list = D.list_instruments(instruments=inst_dict, as_list=True)
    return sorted(inst_list)


def get_instruments_for_market(market: str) -> list[str]:
    """Read instruments/{market}.txt from qlib_dir and return symbols (uppercase qlib format).

    Falls back to empty list if the file doesn't exist.
    """
    init_qlib_once()
    s = Settings()
    f = s.qlib_data_dir / "instruments" / f"{market}.txt"
    if not f.is_file():
        return []
    out = []
    for line in f.read_text(encoding="utf-8").splitlines():
        sym = line.split("\t")[0].strip()
        if sym and not sym.startswith("#"):
            out.append(sym)
    return out


def get_calendar_info() -> tuple[date, int]:
    """Return (last_trading_day, total_calendar_size) for the configured market."""
    init_qlib_once()
    cal = D.calendar(freq="day")
    if not len(cal):
        raise DependencyError("empty trading calendar", code="calendar_empty")
    return pd.Timestamp(cal[-1]).date(), int(len(cal))


def get_market_with_names(market: str) -> list[dict]:
    """Like get_csi300_with_names but for any market. Returns [{symbol, name}, ...]."""
    import json
    init_qlib_once()
    symbols = get_instruments_for_market(market)

    # qlib_adapter.py -> core/ -> app/ -> backend/ -> <repo root>
    project_root = Path(__file__).resolve().parents[3]
    cache_path = project_root / "production" / "cn_names_cache.json"
    name_map: dict[str, str] = {}
    if cache_path.is_file():
        try:
            blob = json.loads(cache_path.read_text(encoding="utf-8"))
            raw = blob.get("map", {}) or {}
            for sym in symbols:
                bare = sym[2:] if sym.startswith(("SH", "SZ")) else sym
                name_map[sym] = raw.get(bare, "")
        except Exception:
            pass

    # ETF names: fall back to production/etf_names.json since cn_names_cache is A-share only.
    etf_names_path = project_root / "production" / "etf_names.json"
    if etf_names_path.is_file():
        try:
            etf_map = json.loads(etf_names_path.read_text(encoding="utf-8"))
            for sym in symbols:
                if not name_map.get(sym):
                    name_map[sym] = etf_map.get(sym, "")
        except Exception:
            pass

    return [{"symbol": sym, "name": name_map.get(sym, "")} for sym in symbols]


def get_csi300_with_names() -> list[dict]:
    """Returns [{symbol, name}] for CSI300 with Chinese names from production cache file."""
    return get_market_with_names("csi300")


import re as _re

# Internal point-in-time snapshots written per-week by
# production/pit_constituents.py (e.g. "csi800_pit_2026-05-22"). These are
# wired into qlib's instruments dir so the training pipeline can pin a
# survivorship-bias-free universe, but they're not meaningful as a
# user-facing universe choice — hide them from the data-source picker.
_INTERNAL_INSTRUMENTS_PATTERN = _re.compile(r"^[a-z]+\d*_pit_\d{4}-\d{2}-\d{2}$", _re.IGNORECASE)


def list_available_markets() -> list[dict]:
    """Scan instruments/ dir for *.txt files. Return [{name, count, label}, ...].

    Filters out:
      - `all.txt` — qlib's full-universe meta file
      - `*_pit_YYYY-MM-DD.txt` — internal weekly PIT snapshots
    """
    init_qlib_once()
    s = Settings()
    inst_dir = s.qlib_data_dir / "instruments"
    if not inst_dir.is_dir():
        return []
    # Label map; unknown markets get name as label
    labels = {
        "csi300": "沪深300",
        "csi500": "中证500",
        "etfs": "热门ETF",
        "custom": "自定义",
        "all": "全部",
    }
    out = []
    for txt in sorted(inst_dir.glob("*.txt")):
        name = txt.stem
        if name == "all":
            continue  # exclude the union meta file from user-facing list
        if _INTERNAL_INSTRUMENTS_PATTERN.match(name):
            continue  # internal PIT snapshot, not a user-selectable universe
        try:
            with txt.open("r", encoding="utf-8") as fh:
                count = sum(1 for ln in fh if ln.strip())
        except Exception:
            count = 0
        out.append({"name": name, "label": labels.get(name, name), "count": count})
    return out


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


def load_pred(
    recorder_id: str,
    experiment_name: str = "daily_cn_fresh",
    series_only: bool = False,
) -> "pd.DataFrame | pd.Series":
    """Load pred.pkl from a recorder.

    Returns the original object (DataFrame for rolling_v2_ensemble with score
    + consensus + per-model base columns, Series for older single-column
    formats). Pass `series_only=True` to coerce a DataFrame to its score
    Series (legacy behaviour; use when the caller only needs the unified
    score and not the per-model base columns).
    """
    init_qlib_once()
    exp = R.get_exp(experiment_name=experiment_name)
    rec = exp.get_recorder(recorder_id=recorder_id)
    pred = rec.load_object("pred.pkl")
    if series_only and isinstance(pred, pd.DataFrame):
        pred = pred["score"]
    return pred


def list_recorder_infos(experiment_name: str) -> dict[str, dict]:
    """Return {recorder_id: info_dict} for an experiment's recorders (info carries
    'name'/'start_time'). Empty dict if the experiment is missing/unreadable —
    callers iterate and skip, so a bad experiment is simply absent."""
    init_qlib_once()
    try:
        recs = R.list_recorders(experiment_name=experiment_name)
    except Exception:
        return {}
    return {rid: (rec.info or {}) for rid, rec in recs.items()}


def get_recorder_run_name(recorder_id: str, experiment_name: str) -> str:
    """Recorder's human run name (info['name']), falling back to the id prefix."""
    init_qlib_once()
    try:
        rec = R.get_recorder(recorder_id=recorder_id, experiment_name=experiment_name)
        return rec.info.get("name", recorder_id[:8])
    except Exception:
        return recorder_id[:8]


def fetch_open_to_open_labels(symbols: list[str], start: str, end: str) -> pd.Series:
    """Open-to-open forward label `Ref($open,-2)/Ref($open,-1)-1` per
    (datetime, instrument) — the label the eval scorecard joins against pred.
    Normalizes qlib's native (instrument, datetime) index to (datetime, instrument).
    Returns an empty Series named 'y' if qlib yields nothing."""
    init_qlib_once()
    df = D.features(
        instruments=symbols,
        fields=["Ref($open, -2) / Ref($open, -1) - 1"],
        start_time=start,
        end_time=end,
    )
    if df is None or df.empty:
        return pd.Series(dtype="float64", name="y")
    df.columns = ["y"]
    s = df["y"]
    if list(s.index.names) != ["datetime", "instrument"]:
        s.index = s.index.set_names(["instrument", "datetime"])
        s = s.swaplevel().sort_index()
    return s
