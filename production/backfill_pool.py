# production/backfill_pool.py
"""Pool per-fold OOF predictions across a date range into one continuous
(datetime,instrument)->score series for long-window backtesting."""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS = ("lgbm", "alstm", "tra")
HORIZONS = ("1d", "5d", "20d")


def assemble_score(base: pd.DataFrame, ewma_alpha: float = 0.5) -> pd.DataFrame:
    """De-dup (keep last), score = -mean(rank over 1d+5d cols), per datetime.
    EWMA-smooth score per instrument (alpha=1.0 disables smoothing)."""
    base = base[~base.index.duplicated(keep="last")].sort_index()
    base.index = base.index.set_names(["datetime", "instrument"])
    score_cols = [c for c in base.columns if not c.endswith("_20d")]
    ranks = base[score_cols].groupby(level="datetime").rank(ascending=False, method="min")
    base["score"] = -ranks.mean(axis=1, skipna=True)
    if ewma_alpha < 1.0:
        base["score"] = (base["score"].groupby(level="instrument")
                         .transform(lambda s: s.ewm(alpha=ewma_alpha).mean()))
    return base


def _rec_name(r) -> str:
    info = getattr(r, "info", {})
    if isinstance(info, dict) and info.get("name"):
        return info["name"]
    return getattr(r, "name", "") or ""


def pool_range(start: date, end: date, *, models=DEFAULT_MODELS,
               config_path: str = "production/configs/rolling_ensemble.yaml",
               out_path: str | None = None) -> Path:
    """Load every <model>_<h>_<fold> recorder with fold in [start,end], concat each
    model_horizon column across folds, assemble score, write one long pred pickle."""
    from qlib.workflow import R
    from production.rolling_train import load_config, init_qlib
    cfg = load_config(Path(config_path))
    init_qlib(cfg)
    exp = R.get_exp(experiment_name=cfg.experiment_name)
    recs = exp.list_recorders()
    recs = list(recs.values()) if isinstance(recs, dict) else recs

    pat = re.compile(r"^(%s)_(1d|5d|20d)_(\d{4}-\d{2}-\d{2})$" % "|".join(models))
    cols: dict[str, list[pd.Series]] = {}
    for r in recs:
        m = pat.match(_rec_name(r))
        if not m:
            continue
        fold = date.fromisoformat(m.group(3))
        if not (start <= fold <= end):
            continue
        model, h = m.group(1), m.group(2)
        try:
            s = r.load_object(f"pred_{h}.pkl")
        except Exception:
            continue
        if isinstance(s, pd.DataFrame):
            s = s["score"] if "score" in s.columns else s.iloc[:, 0]
        cols.setdefault(f"{model}_{h}", []).append(s)

    if not cols:
        raise SystemExit("no matching recorders in range")
    merged = {k: pd.concat(v).sort_index() for k, v in cols.items()}
    base = pd.concat([s.rename(k) for k, s in merged.items()], axis=1).sort_index()
    base = assemble_score(base)
    out = Path(out_path or REPO_ROOT / "production" / "reports" /
               f"oof_{start.isoformat()}_{end.isoformat()}.pkl")
    out.parent.mkdir(parents=True, exist_ok=True)
    base.to_pickle(out)
    print(f"wrote {out} rows={len(base)} span={base.index.get_level_values('datetime').min()}"
          f"..{base.index.get_level_values('datetime').max()}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--models", default="lgbm,alstm,tra")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pool_range(date.fromisoformat(a.start), date.fromisoformat(a.end),
               models=tuple(a.models.split(",")), out_path=a.out)
