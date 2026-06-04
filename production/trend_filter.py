# production/trend_filter.py
"""Trend-aware ranking gate for the A-share short-term selector.

The ensemble (LGBM+ALSTM) has a REVERSAL BIAS: its top-ranked names are
disproportionately stocks in clear downtrends (空头排列 / "falling knives").
This module provides a trailing, no-lookahead gate that *removes* (drops)
non-trending names from the score series BEFORE the rebalance policy sees
them, so `_equal_top_k` / `Banded` can never select a falling knife.

It is engine-agnostic: `apply_trend_filter` simply returns a score Series with
the non-passing (datetime, instrument) entries removed. The backtest engine in
``production/backtest/engine.py`` slices scores per date with
``s.xs(d, level="datetime")``, so removed entries are invisible to selection;
no change to engine.py / rebalance.py is required.

No-lookahead guarantee
----------------------
Every feature is TRAILING. The verdict at date D depends only on closes
<= D (the decision is made after the close of D, before the next open).
``compute_trend_features`` uses backward-looking rolling windows
(``.rolling(w)`` = closes[i-w+1 .. i]) and a backward shift for momentum
(``close / close.shift(20) - 1``). There is no forward reference anywhere.

Gate modes
----------
- ``"none"``   : passthrough (baseline; returns scores unchanged).
- ``"soft"``   : keep iff ``close > MA20`` (above the 20-day mean).
- ``"strict"`` : keep iff ``MA5 > MA10 > MA20`` AND ``momentum_20 > 0`` AND
                 ``close > MA20`` (clean multi-head 多头排列, positive 20d
                 momentum, price above its mean). The extra ``close > MA20``
                 term makes ``strict`` a strict subset of ``soft``.

Warm-up rows (incomplete trailing windows -> NaN features) cannot CONFIRM a
trend, so a real gate drops them by default (``drop_warmup=True``). Pass
``drop_warmup=False`` to keep warm-up rows (treated as passing) if you would
rather not lose early dates.
"""
from __future__ import annotations

# --- sys.path fixup: same pattern as production/backtest/data.py ---
# Ensure the installed qlib (compiled) wins over the uncompiled ./qlib source
# when this module is imported under `-m production...`. Harmless when qlib is
# not used (the unit tests never import qlib).
import sys as _sys
import sysconfig as _sysconfig

_PURELIB = _sysconfig.get_paths().get("purelib")
if _PURELIB and _PURELIB not in _sys.path[:1]:
    _sys.path.insert(0, _PURELIB)
# --- end fixup ---

import numpy as np
import pandas as pd

# qlib expression form of the trailing trend features (used by _eval_trend.py
# when loading real prices via QlibDataLoader). MUST stay trailing / no Ref(-k).
TREND_FEATURE_EXPRS: list[str] = [
    "$close",
    "Mean($close, 5)",
    "Mean($close, 10)",
    "Mean($close, 20)",
    "Mean($close, 60)",
    "$close / Ref($close, 20) - 1",   # 20d momentum (Ref +20 = 20 days ago)
]
TREND_FEATURE_NAMES: list[str] = [
    "close_gt_ma20",   # close > MA20
    "close_gt_ma60",   # close > MA60
    "ma_aligned",      # MA5 > MA10 > MA20  (多头排列)
    "momentum",        # close/close[-20] - 1
    "ma20",            # raw MA20 (diagnostics / tests)
    "ma60",            # raw MA60 (diagnostics)
]

VALID_MODES = ("none", "soft", "strict")

# columns that mark a frame as ALREADY a computed feature frame
_FEATURE_MARKERS = ("close_gt_ma20", "ma_aligned", "momentum")


# --------------------------------------------------------------------------- #
# feature computation (trailing, no lookahead)                                 #
# --------------------------------------------------------------------------- #
def _as_close_series(prices) -> pd.Series:
    """Coerce a (datetime, instrument) close panel into a sorted Series."""
    if isinstance(prices, pd.DataFrame):
        col = "close" if "close" in prices.columns else prices.columns[0]
        s = prices[col]
    else:
        s = prices
    s = s.copy()
    if list(s.index.names) and s.index.names[0] == "instrument":
        s = s.swaplevel().sort_index()
    s.index = s.index.set_names(["datetime", "instrument"])
    return s.sort_index()


def compute_trend_features(prices) -> pd.DataFrame:
    """Compute trailing trend features from a (datetime, instrument) close panel.

    Parameters
    ----------
    prices : Series or 1-col DataFrame, MultiIndex (datetime, instrument)
        Daily close prices.

    Returns
    -------
    DataFrame indexed by (datetime, instrument) with columns
    ``TREND_FEATURE_NAMES``. Boolean-flag columns are float (1.0/0.0) with NaN
    during the per-instrument warm-up so callers can distinguish "fails" from
    "unknown".
    """
    close = _as_close_series(prices).rename("close")

    # Pivot to a wide (datetime x instrument) frame so rolling is a single
    # vectorized, unambiguously TRAILING op per column. Avoids version-fragile
    # groupby.apply semantics. `wide` rows are sorted by datetime ascending.
    wide = close.unstack("instrument").sort_index()

    def _ma(w: int) -> pd.DataFrame:
        # rolling(w) over the datetime axis = closes[i-w+1 .. i] (trailing).
        return wide.rolling(w, min_periods=w).mean()

    ma5, ma10, ma20, ma60 = _ma(5), _ma(10), _ma(20), _ma(60)
    mom = wide / wide.shift(20) - 1.0   # backward shift -> 20d momentum

    aligned = (ma5 > ma10) & (ma10 > ma20)
    valid_align = ma5.notna() & ma10.notna() & ma20.notna()

    def _flag(cond: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
        return cond.astype(float).where(valid, np.nan)

    flags = {
        "close_gt_ma20": _flag(wide > ma20, ma20.notna()),
        "close_gt_ma60": _flag(wide > ma60, ma60.notna()),
        "ma_aligned": _flag(aligned, valid_align),
        "momentum": mom,
        "ma20": ma20,
        "ma60": ma60,
    }
    # Reindex each wide frame back onto the ORIGINAL long (datetime, instrument)
    # index. This (a) keeps NaNs from warm-up so callers can tell "fail" from
    # "unknown", and (b) avoids fabricating rows for (date,inst) pairs that never
    # traded in a ragged panel — and sidesteps stack()'s pandas-version churn.
    dt = close.index.get_level_values("datetime")
    inst = close.index.get_level_values("instrument")
    row_pos = wide.index.get_indexer(dt)       # datetime -> wide row position
    col_pos = wide.columns.get_indexer(inst)   # instrument -> wide col position
    cols = {}
    for name in TREND_FEATURE_NAMES:
        # vectorized gather: value at (datetime row, instrument col) per pair
        cols[name] = pd.Series(
            flags[name].to_numpy()[row_pos, col_pos],
            index=close.index,
            name=name,
        )
    df = pd.DataFrame(cols)
    df.index = df.index.set_names(["datetime", "instrument"])
    return df[TREND_FEATURE_NAMES].sort_index()


# --------------------------------------------------------------------------- #
# the gate                                                                     #
# --------------------------------------------------------------------------- #
def _is_feature_frame(obj) -> bool:
    return isinstance(obj, pd.DataFrame) and all(m in obj.columns for m in _FEATURE_MARKERS)


def _pass_mask(features: pd.DataFrame, mode: str, *, drop_warmup: bool) -> pd.Series:
    """Boolean Series over `features.index`: True == passes the gate."""
    if mode == "none":
        return pd.Series(True, index=features.index)

    if mode == "soft":
        flag = features["close_gt_ma20"]
    elif mode == "strict":
        # MA5>MA10>MA20 (ma_aligned) AND momentum>0 AND close>MA20.
        # close>MA20 is implied by intent and GUARANTEES strict is a subset of
        # soft (strict can never keep a name the soft gate would drop).
        aligned = features["ma_aligned"]
        mom_pos = (features["momentum"] > 0).astype(float).where(
            features["momentum"].notna(), np.nan)
        above = features["close_gt_ma20"]  # already nan-aware float
        # AND of nan-aware floats: nan if any unknown, else 1.0/0.0
        flag = aligned * mom_pos * above
    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"unknown mode {mode!r}; expected one of {VALID_MODES}")

    if drop_warmup:
        # NaN (warm-up / unknown) -> treated as FAIL (cannot confirm trend).
        return flag.fillna(0.0) > 0.5
    # keep warm-up rows: NaN -> treated as PASS.
    return flag.fillna(1.0) > 0.5


def apply_trend_filter(
    scores: pd.Series,
    prices_or_features,
    *,
    mode: str = "soft",
    drop_warmup: bool = True,
) -> pd.Series:
    """Remove non-trending (datetime, instrument) entries from `scores`.

    Parameters
    ----------
    scores : Series, MultiIndex (datetime, instrument) -> score (higher = better).
    prices_or_features :
        Either a precomputed trend-feature DataFrame (as produced by
        ``compute_trend_features``; detected by its marker columns) OR a raw
        close-price panel (Series / 1-col DataFrame) from which features are
        computed internally.
    mode : {"none", "soft", "strict"}
        Gate strength. See module docstring.
    drop_warmup : bool, default True
        If True, rows with incomplete trailing windows (NaN features) are
        dropped (a gate cannot confirm a trend it cannot see). If False, such
        rows are kept (treated as passing).

    Returns
    -------
    Series : `scores` with non-passing entries removed. The returned index is
    always a SUBSET of `scores.index` (the gate never adds names). Score values
    of surviving rows are unchanged.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {VALID_MODES}")
    if not isinstance(scores, pd.Series):
        raise TypeError("scores must be a pandas Series indexed by (datetime, instrument)")
    if scores.empty:
        return scores.copy()
    if mode == "none":
        return scores.copy()

    features = prices_or_features if _is_feature_frame(prices_or_features) \
        else compute_trend_features(prices_or_features)

    # normalise the feature index names to match scores
    if list(features.index.names) and features.index.names[0] == "instrument":
        features = features.swaplevel().sort_index()
    features = features.copy()
    features.index = features.index.set_names(["datetime", "instrument"])

    mask = _pass_mask(features, mode, drop_warmup=drop_warmup)
    passing_idx = mask.index[mask.values]

    # Keep only score rows that (a) have a feature row AND (b) pass the gate.
    # Rows in `scores` with NO feature row at all are dropped when drop_warmup
    # (we cannot confirm their trend); kept when not drop_warmup.
    keep = scores.index.intersection(passing_idx)
    if not drop_warmup:
        missing = scores.index.difference(features.index)
        keep = keep.union(missing)
    out = scores.loc[scores.index.isin(keep)]
    return out.rename(scores.name)


# --------------------------------------------------------------------------- #
# convenience: classify a single close vector (live trader screen)             #
# --------------------------------------------------------------------------- #
def is_downtrend(close_vec: np.ndarray) -> bool:
    """True if `close_vec` (chronological) ends below its trailing MA20.

    Mirrors the falling-knife test used in production/_pick_trader.py. Used by
    the live sanity check to count how many raw top-N picks are falling knives.
    Returns False if there is insufficient history (<20 closes).
    """
    cl = np.asarray(close_vec, dtype=float)
    if cl.size < 20 or not np.isfinite(cl[-1]):
        return False
    ma20 = cl[-20:].mean()
    return bool(cl[-1] < ma20)
