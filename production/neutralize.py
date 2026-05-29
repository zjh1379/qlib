"""Cross-sectional score neutralization (sector demean + size regression)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def neutralize(scores: pd.Series, sector: pd.Series | None = None,
               size: pd.Series | None = None) -> pd.Series:
    """Per-datetime: subtract sector mean, then regress out size (rank).
    `sector`/`size` are instrument-indexed Series. Returns same-index Series."""
    name = scores.name
    parts = []
    for d, g in scores.groupby(level="datetime"):
        x = g.copy()
        inst = x.index.get_level_values("instrument")
        if sector is not None:
            sec = pd.Series(sector.reindex(inst).values, index=x.index)
            grp_mean = x.groupby(sec.values).transform("mean")
            x = x - pd.Series(grp_mean.values, index=x.index)
        if size is not None:
            sz = pd.Series(size.reindex(inst).values, index=x.index).rank()
            zc = sz - sz.mean()
            xc = x - x.mean()
            denom = float((zc * zc).sum())
            beta = float((xc * zc).sum() / denom) if denom > 0 else 0.0
            x = x - beta * zc
        parts.append(x)
    return pd.concat(parts).rename(name)
