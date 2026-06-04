"""Regenerate the live served prediction with the FACTOR-LGBM model.

Why: the factor LGBM (Alpha158 + 6 short-term factors) beats baseline LGBM by
+12pp net CAGR (see 2026-06-03-shortterm-factors-results.md). This swaps the
factor LGBM predictions into the latest live 2-model pred and writes a fresh
recorder so the backend (get_latest_recorder_id = newest recorder with pred.pkl)
serves the factor model.

Base = examples/mlruns/pred_2026-06-02.pkl (5 dates 2026-05-27..06-02; already
has alstm_* + lgbm_* + score + consensus). We replace lgbm_{1d,5d,20d} with the
factor-LGBM per-horizon predictions trained today (run-once --features shortterm
--end-date 2026-06-02), recompute score (-rank_avg over 1d+5d cols, v9) and
consensus, and save to a NEW recorder 'ensemble_factor_2026-06-02'.

Run: F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m production._regen_live_factor
"""
import sys as _sys
import sysconfig as _sysconfig
_P = _sysconfig.get_paths().get("purelib")
if _P and _P not in _sys.path[:1]:
    _sys.path.insert(0, _P)
try:
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from pathlib import Path
import numpy as np
import pandas as pd

EXP = "rolling_v2_ensemble"
BASE = "examples/mlruns/pred_2026-06-02.pkl"
NEW_RECORDER = "ensemble_factor_2026-06-02"


def _nm(r) -> str:
    i = getattr(r, "info", {})
    return (i.get("name") if isinstance(i, dict) else getattr(r, "name", "")) or ""


def _newest(recs, name):
    cand = [r for r in recs if _nm(r) == name]
    return max(cand, key=lambda r: r.info["start_time"]) if cand else None


def _recompute(df: pd.DataFrame) -> pd.DataFrame:
    score_cols = [c for c in df.columns
                  if (c.endswith("_1d") or c.endswith("_5d"))
                  and not c.startswith(("expected_", "composite_"))]
    ranks = df[score_cols].groupby(level="datetime").rank(ascending=False, method="min")
    df["score"] = -ranks.mean(axis=1, skipna=True)
    five = [c for c in df.columns if c.endswith("_5d")
            and not c.startswith(("expected_", "composite_"))]
    signs = np.sign(df[five].fillna(0).values)
    df["consensus"] = np.abs(signs.sum(axis=1)) / max(1, signs.shape[1])
    return df


def main() -> int:
    from qlib.workflow import R
    from production.rolling_train import load_config, init_qlib
    cfg = load_config(Path("production/configs/rolling_ensemble.yaml"))
    init_qlib(cfg)
    exp = R.get_exp(experiment_name=EXP)
    recs = exp.list_recorders()
    recs = list(recs.values()) if isinstance(recs, dict) else recs

    base = pd.read_pickle(BASE).copy()
    base.index = base.index.set_names(["datetime", "instrument"])
    latest = base.index.get_level_values("datetime").max()
    before_top = (base.xs(latest, level="datetime")["score"]
                  .sort_values(ascending=False).head(5).index.tolist())

    # factor lgbm per-horizon preds (newest lgbm_{h}_2026-06-02 = trained today)
    cover = {}
    for h in ("1d", "5d", "20d"):
        r = _newest(recs, f"lgbm_{h}_2026-06-02")
        if r is None:
            print(f"MISSING factor recorder lgbm_{h}_2026-06-02"); return 1
        s = r.load_object(f"pred_{h}.pkl")
        if isinstance(s, pd.DataFrame):
            s = s["score"] if "score" in s.columns else s.iloc[:, 0]
        s.index = s.index.set_names(["datetime", "instrument"])
        s = s.reindex(base.index)
        cover[h] = float(s.notna().mean())
        base[f"lgbm_{h}"] = s.combine_first(base[f"lgbm_{h}"])  # prefer factor
    print("factor-lgbm coverage of base index:", {k: f"{v:.0%}" for k, v in cover.items()})

    base = _recompute(base)
    after_top = (base.xs(latest, level="datetime")["score"]
                 .sort_values(ascending=False).head(5).index.tolist())
    print(f"latest={latest.date()}  rows={len(base)}")
    print("top5 BEFORE (baseline-lgbm):", before_top)
    print("top5 AFTER  (factor-lgbm) :", after_top)

    with R.start(experiment_name=EXP, recorder_name=NEW_RECORDER):
        R.save_objects(**{"pred.pkl": base})
    print(f"WROTE new recorder '{NEW_RECORDER}' with factor pred.pkl in {EXP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
