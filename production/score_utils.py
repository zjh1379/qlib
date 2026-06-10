"""Canonical score reconstruction — the single home for "what the model says".

`rebuild_2model` rebuilds the champion ensemble score (factor-LGBM + ALSTM) from
saved OOF prediction frames via the production rank-blend (`assemble_score`). It is
imported by the research eval scripts AND by the intraday execution simulator, so it
lives here as stable, tested code rather than inside a throwaway eval runner.

Pure pandas; `assemble_score` is imported lazily so importing this module never
pulls in qlib."""
from __future__ import annotations
import numpy as np
import pandas as pd


def score_of(df) -> pd.Series:
    """Normalize a saved prediction (Series or DataFrame) to a clean score Series
    named 'score' with a (datetime, instrument) MultiIndex, NaNs dropped."""
    if isinstance(df, pd.Series):
        s = df
    else:
        s = df["score"] if "score" in df.columns else df.iloc[:, 0]
    s = s.copy()
    s.index = s.index.set_names(["datetime", "instrument"])
    return s.rename("score").dropna()


def rebuild_2model(lgbm_df: pd.DataFrame, two_df: pd.DataFrame) -> pd.Series:
    """Rebuild the canonical ensemble score from a given LGBM's 1d/5d/20d columns
    + the baseline ALSTM 1d/5d/20d columns, via assemble_score (the production
    -mean(rank over non-_20d cols) blend). Lets us swap in factor-LGBM while
    keeping the same ALSTM, for an apples-to-apples 2-model comparison."""
    from production.backfill_pool import assemble_score
    lg = lgbm_df[[c for c in lgbm_df.columns if c.startswith("lgbm_")]]
    al = two_df[[c for c in two_df.columns if c.startswith("alstm_")]]
    base = pd.concat([lg, al], axis=1).sort_index()
    base.index = base.index.set_names(["datetime", "instrument"])
    scored = assemble_score(base)
    return score_of(scored)


def calmar(m: dict) -> float:
    """Calmar ratio (net CAGR / |max drawdown|) from a net_metrics dict; nan-safe."""
    dd, cagr = m.get("max_drawdown"), m.get("net_cagr")
    if dd is None or not np.isfinite(dd) or abs(dd) < 1e-12:
        return float("nan")
    return cagr / abs(dd)
