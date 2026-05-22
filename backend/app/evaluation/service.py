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


# Cache helpers — wired in Task 3 once evaluate_recorder exists.
def _cache_has(recorder_id: str) -> bool:
    """Return True iff evaluate_recorder has been called for this recorder
    since process startup."""
    return False


def _cache_quick_look(recorder_id: str) -> tuple[float | None, float | None, bool | None]:
    """Pull (ic_mean, ir, acceptance_passed) from cache if present."""
    return (None, None, None)
