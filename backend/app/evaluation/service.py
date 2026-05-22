"""Evaluation service — load recorders, compute scorecards, compare.

All metric math is delegated to production/metrics.py + production/validate_acceptance.py
(the rolling_train pipeline uses the same helpers, so eval numbers always
match what was computed at train time).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from app.core.config import Settings
from app.core.qlib_adapter import init_qlib_once
from app.evaluation.schemas import RecorderSummary

# Add the repo root so we can import production.metrics / production.validate_acceptance
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))


def list_recorders_with_summary() -> list[RecorderSummary]:
    """Enumerate all qlib recorders across all experiments and return a
    lightweight summary for each. Cheap — does NOT load pred.pkl."""
    init_qlib_once()
    from qlib.workflow import R

    out: list[RecorderSummary] = []
    # List experiments (excludes the deleted-trash sentinel)
    exp_ids = _list_experiment_ids()
    for exp_id in exp_ids:
        exp_name = _experiment_name(exp_id)
        if exp_name in (None, "Default"):
            continue
        try:
            recs = R.list_recorders(experiment_name=exp_name)
        except Exception:
            continue
        for rec_id, rec in recs.items():
            info = rec.info or {}
            run_name = info.get("name", rec_id[:8])
            created_at = _format_created_at(info.get("start_time"))
            # Stat pred.pkl size cheaply (without loading the dataframe)
            pred_start, pred_end, pred_rows = _peek_pred_pkl(rec_id, exp_name)
            cached = _cache_has(rec_id)
            quick = _cache_quick_look(rec_id) if cached else (None, None, None)
            out.append(
                RecorderSummary(
                    recorder_id=rec_id,
                    experiment=exp_name,
                    run_name=run_name,
                    created_at=created_at,
                    pred_start=pred_start,
                    pred_end=pred_end,
                    pred_rows=pred_rows,
                    has_eval=cached,
                    ic_mean=quick[0],
                    ir=quick[1],
                    acceptance_passed=quick[2],
                )
            )

    # Sort by created_at desc (newest first)
    out.sort(key=lambda s: s.created_at, reverse=True)
    return out


def _format_created_at(start_time) -> str:
    """Normalise the recorder `info['start_time']` into an ISO-8601 string.

    qlib/mlflow returns this field as either a formatted date string
    ('2026-05-09 10:27:21') or as an epoch-millisecond integer depending on
    backend version. Returns '' if the field is missing/unparseable.
    """
    if start_time in (None, "", 0):
        return ""
    try:
        if isinstance(start_time, (int, float)):
            return pd.to_datetime(start_time, unit="ms", utc=True).isoformat()
        # String forms like '2026-05-09 10:27:21'
        return pd.to_datetime(start_time, utc=True).isoformat()
    except Exception:
        return ""


def _list_experiment_ids() -> list[str]:
    """Walk <mlruns_root>/<exp_id>/ and return all valid experiment dir names."""
    settings = Settings()
    root = settings.mlruns_path
    if not root.exists():
        return []
    out = []
    for d in root.iterdir():
        if d.is_dir() and (d / "meta.yaml").exists() and d.name != ".trash":
            out.append(d.name)
    return out


def _experiment_name(exp_id: str) -> str | None:
    """Read mlruns/<exp_id>/meta.yaml and return the experiment name."""
    settings = Settings()
    meta = settings.mlruns_path / exp_id / "meta.yaml"
    if not meta.exists():
        return None
    for line in meta.read_text(encoding="utf-8").splitlines():
        if line.startswith("name:"):
            return line.split(":", 1)[1].strip()
    return None


def _peek_pred_pkl(rec_id: str, exp_name: str) -> tuple[str | None, str | None, int | None]:
    """Load the recorder's pred.pkl just enough to return date range + row count.
    Returns (None, None, None) on missing/unreadable files."""
    settings = Settings()
    # Find the artifact path. qlib mlflow layout: <mlruns_root>/<exp_id>/<rec_id>/artifacts/
    exp_id = _find_exp_id_for_name(exp_name)
    if exp_id is None:
        return None, None, None
    artifacts = settings.mlruns_path / exp_id / rec_id / "artifacts"
    # Most common name is pred.pkl; for sub-horizon recorders it's pred_<N>.pkl
    candidates = [
        artifacts / "pred.pkl",
        artifacts / "pred_1d.pkl",
        artifacts / "pred_5d.pkl",
        artifacts / "pred_20d.pkl",
    ]
    for c in candidates:
        if c.exists():
            try:
                df = pd.read_pickle(c)
                if isinstance(df, pd.Series):
                    df = df.to_frame()
                dates = df.index.get_level_values(0)
                return (
                    str(pd.Timestamp(dates.min()).date()),
                    str(pd.Timestamp(dates.max()).date()),
                    int(df.shape[0]),
                )
            except Exception:
                continue
    return None, None, None


def _find_exp_id_for_name(exp_name: str) -> str | None:
    for exp_id in _list_experiment_ids():
        if _experiment_name(exp_id) == exp_name:
            return exp_id
    return None


import functools
from datetime import datetime, timezone

from qlib.workflow import R as _R
from qlib.data import D as _D

from app.evaluation.schemas import (
    AcceptanceResult,
    EvalResult,
    RegimeMetrics,
    ScorecardData,
)

# Spec §8 multi-regime segments. evaluate_recorder filters to those overlapping
# the recorder's prediction window; for recent-only recorders (e.g. 2025+) we
# always synthesize a "Recent" segment covering the full window.
_REGIME_SEGMENTS: list[tuple[str, str, str]] = [
    ("2018 Bear", "2018-01-01", "2018-12-31"),
    ("2019-20 Recovery", "2019-01-01", "2020-02-28"),
    ("2020-21 COVID liquidity", "2020-03-01", "2021-02-28"),
    ("2021-22 High vol", "2021-03-01", "2022-10-31"),
    ("2022-24 AI rally", "2022-11-01", "2024-12-31"),
]

# Sidecar maps populated by evaluate_recorder() so list view can show
# "已评估" status + quick-look IC/IR. Cleared on force_refresh.
_CACHE_SEEN: set[str] = set()
_CACHE_RESULTS: dict[str, EvalResult] = {}


def evaluate_recorder(
    recorder_id: str,
    top_k: int = 30,
    cost_bps: float = 10.0,
    force_refresh: bool = False,
) -> EvalResult:
    """Run the full 8-metric scorecard + regime split + acceptance check.

    Results cached in-process by (recorder_id, top_k, cost_bps).
    `force_refresh=True` clears the entire cache and recomputes.
    """
    if force_refresh:
        _evaluate_cached.cache_clear()
        _CACHE_SEEN.clear()
        _CACHE_RESULTS.clear()
    result = _evaluate_cached(recorder_id, top_k, cost_bps)
    _CACHE_SEEN.add(recorder_id)
    _CACHE_RESULTS[recorder_id] = result
    return result


@functools.lru_cache(maxsize=32)
def _evaluate_cached(recorder_id: str, top_k: int, cost_bps: float) -> EvalResult:
    """Heavy path: load pred.pkl, fetch labels, compute scorecard + regimes."""
    from production.metrics import compute_scorecard, regime_split
    from production.validate_acceptance import check_acceptance

    init_qlib_once()
    exp_name = _experiment_for_recorder(recorder_id)
    if exp_name is None:
        raise ValueError(f"recorder {recorder_id} not found in any experiment")

    pred = _load_pred_as_series(recorder_id, exp_name)
    if pred.empty:
        raise ValueError(f"recorder {recorder_id}: pred.pkl empty or missing")

    labels = _fetch_open_to_open_labels(pred)

    # Run scorecard
    scorecard_dict = compute_scorecard(pred, labels, top_k=top_k, bps=cost_bps)
    scorecard = ScorecardData(**scorecard_dict)

    # Regime split — only segments overlapping the prediction range, plus a Recent catch-all
    overlapping = _overlapping_regimes(pred)
    regimes_raw = regime_split(pred, labels, [(s, e) for _, s, e in overlapping])
    regimes: list[RegimeMetrics] = []
    for label_name, start, end in overlapping:
        key = f"{start}__{end}"
        seg = regimes_raw.get(key)
        if seg is None:
            # Segment had no overlapping data; skip rather than emit zeros.
            continue
        mask = (
            (pred.index.get_level_values("datetime") >= pd.Timestamp(start))
            & (pred.index.get_level_values("datetime") <= pd.Timestamp(end))
        )
        regimes.append(
            RegimeMetrics(
                label=label_name,
                start=start,
                end=end,
                sample_size=int(mask.sum()),
                scorecard=ScorecardData(**seg),
            )
        )

    regime_irs = {r.label: r.scorecard.ir for r in regimes}
    acceptance_dict = check_acceptance(scorecard_dict, regime_irs)
    acceptance = AcceptanceResult(**acceptance_dict)

    # Compute the actual evaluation window from labels (post-join)
    joined_dates = (
        pd.concat([pred.rename("p"), labels.rename("y")], axis=1).dropna()
        .index.get_level_values("datetime")
    )
    window_start = str(joined_dates.min().date()) if len(joined_dates) else ""
    window_end = str(joined_dates.max().date()) if len(joined_dates) else ""

    rec = _R.get_recorder(recorder_id=recorder_id, experiment_name=exp_name)
    run_name = rec.info.get("name", recorder_id[:8])

    return EvalResult(
        recorder_id=recorder_id,
        experiment=exp_name,
        run_name=run_name,
        computed_at=datetime.now(timezone.utc).isoformat(),
        window_start=window_start,
        window_end=window_end,
        sample_size=len(joined_dates),
        top_k=top_k,
        cost_bps=cost_bps,
        scorecard=scorecard,
        regimes=regimes,
        acceptance=acceptance,
    )


def _experiment_for_recorder(recorder_id: str) -> str | None:
    """Find which experiment owns this recorder_id."""
    for exp_id in _list_experiment_ids():
        rec_dir = Settings().mlruns_path / exp_id / recorder_id
        if rec_dir.is_dir():
            return _experiment_name(exp_id)
    return None


def _load_pred_as_series(recorder_id: str, exp_name: str) -> pd.Series:
    """Load pred.pkl (or pred_5d.pkl etc.) as a 1-col Series indexed by (datetime, instrument).
    For multi-column DataFrames (ensemble output), uses the 'score' column."""
    exp_id = _find_exp_id_for_name(exp_name)
    if exp_id is None:
        return pd.Series(dtype="float64")
    artifacts = Settings().mlruns_path / exp_id / recorder_id / "artifacts"
    candidates = [artifacts / "pred.pkl", artifacts / "pred_5d.pkl",
                  artifacts / "pred_1d.pkl", artifacts / "pred_20d.pkl"]
    for c in candidates:
        if not c.exists():
            continue
        df = pd.read_pickle(c)
        if isinstance(df, pd.Series):
            return _ensure_index(df.rename("score"))
        if "score" in df.columns:
            return _ensure_index(df["score"])
        return _ensure_index(df.iloc[:, 0])
    return pd.Series(dtype="float64")


def _ensure_index(s: pd.Series) -> pd.Series:
    if s.index.names != ["datetime", "instrument"]:
        s.index = s.index.set_names(["datetime", "instrument"])
    return s.sort_index()


def _fetch_open_to_open_labels(pred: pd.Series) -> pd.Series:
    """Pull Ref($open, -2) / Ref($open, -1) - 1 from qlib for the same
    (date, symbol) range as pred. Returns a Series with the same index layout."""
    symbols = sorted(pred.index.get_level_values("instrument").unique().tolist())
    dates = pred.index.get_level_values("datetime")
    start = (pd.Timestamp(dates.min()) - pd.Timedelta(days=5)).date()
    end = (pd.Timestamp(dates.max()) + pd.Timedelta(days=10)).date()
    df = _D.features(
        instruments=symbols,
        fields=["Ref($open, -2) / Ref($open, -1) - 1"],
        start_time=str(start),
        end_time=str(end),
    )
    df.columns = ["y"]
    s = df["y"]
    if s.index.names != ["datetime", "instrument"]:
        # qlib usually returns (instrument, datetime); normalize.
        s.index.names = ["instrument", "datetime"]
        s = s.swaplevel().sort_index()
    return s


def _overlapping_regimes(pred: pd.Series) -> list[tuple[str, str, str]]:
    """Return the spec regime segments whose [start, end] overlap the
    prediction window. Always appends a 'Recent' synthetic segment covering
    the full prediction window so recorders with only 2025+ predictions still
    get something."""
    dates = pred.index.get_level_values("datetime")
    pred_start = pd.Timestamp(dates.min())
    pred_end = pd.Timestamp(dates.max())
    out: list[tuple[str, str, str]] = []
    for label, start_s, end_s in _REGIME_SEGMENTS:
        s = pd.Timestamp(start_s)
        e = pd.Timestamp(end_s)
        if e < pred_start or s > pred_end:
            continue
        out.append((label, start_s, end_s))
    # Catch-all "Recent" segment over the whole prediction range
    out.append(("Recent (full window)", str(pred_start.date()), str(pred_end.date())))
    return out


def _cache_has(recorder_id: str) -> bool:
    """True iff evaluate_recorder has been called for this recorder
    since process startup (or last force_refresh)."""
    return recorder_id in _CACHE_SEEN


def _cache_quick_look(recorder_id: str) -> tuple[float | None, float | None, bool | None]:
    """If cached, return (ic_mean, ir, acceptance_passed) for the list view's quick column."""
    res = _CACHE_RESULTS.get(recorder_id)
    if res is None:
        return (None, None, None)
    return (res.scorecard.ic_mean, res.scorecard.ir, res.acceptance.passed)


def compare_recorders(
    recorder_id_a: str,
    recorder_id_b: str,
    top_k: int = 30,
    cost_bps: float = 10.0,
) -> "CompareResult":
    """Side-by-side eval of 2 recorders + paired t-test on their daily IC series.

    Uses the cached evaluate_recorder() under the hood. If either recorder
    hasn't been evaluated yet, this triggers a (potentially slow) first eval.

    Verdict logic:
        - paired t-test p < 0.05 AND ic_delta > 0  -> "b significantly better"
        - paired t-test p < 0.05 AND ic_delta < 0  -> "a significantly better"
        - otherwise                                -> "no significant difference"
    """
    from production.metrics import paired_ttest
    from app.evaluation.schemas import CompareResult

    result_a = evaluate_recorder(recorder_id_a, top_k=top_k, cost_bps=cost_bps)
    result_b = evaluate_recorder(recorder_id_b, top_k=top_k, cost_bps=cost_bps)

    # Compute daily IC series for both, on the OVERLAPPING date range
    daily_ic_a = _daily_ic_for_recorder(recorder_id_a)
    daily_ic_b = _daily_ic_for_recorder(recorder_id_b)
    if len(daily_ic_a) == 0 or len(daily_ic_b) == 0:
        # No overlapping data — t-test undefined
        t_stat, p_value, significant = float("nan"), 1.0, False
    else:
        try:
            t_stat, p_value = paired_ttest(daily_ic_a, daily_ic_b)
        except Exception:
            t_stat, p_value = float("nan"), 1.0
        significant = p_value < 0.05

    ic_delta = result_b.scorecard.ic_mean - result_a.scorecard.ic_mean
    ir_delta = result_b.scorecard.ir - result_a.scorecard.ir

    if significant and ic_delta > 0:
        verdict = "b significantly better"
    elif significant and ic_delta < 0:
        verdict = "a significantly better"
    else:
        verdict = "no significant difference"

    return CompareResult(
        a=result_a,
        b=result_b,
        paired_t_stat=float(t_stat) if pd.notna(t_stat) else 0.0,
        paired_p_value=float(p_value),
        significant_at_05=bool(significant),
        ic_delta=float(ic_delta),
        ir_delta=float(ir_delta),
        verdict=verdict,
    )


def _daily_ic_for_recorder(recorder_id: str) -> pd.Series:
    """Recompute the daily IC time series for a recorder.

    Returns a Series indexed by date with one IC value per day. Used by
    paired_ttest in compare_recorders.
    """
    exp_name = _experiment_for_recorder(recorder_id)
    if exp_name is None:
        return pd.Series(dtype="float64")
    pred = _load_pred_as_series(recorder_id, exp_name)
    if pred.empty:
        return pd.Series(dtype="float64")
    labels = _fetch_open_to_open_labels(pred)

    import numpy as np
    df = pd.concat([pred.rename("p"), labels.rename("y")], axis=1).dropna()
    return df.groupby(level="datetime").apply(
        lambda g: g["p"].corr(g["y"]) if len(g) > 2 else np.nan
    ).dropna()
