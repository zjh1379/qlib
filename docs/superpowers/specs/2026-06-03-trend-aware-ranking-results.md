# Trend-Aware Ranking · Design + Results

**Date**: 2026-06-03
**Status**: ✅ Complete — code merged-to-worktree, 15/15 unit tests green, long-window eval run, verdict below.
**Owner**: zhu.jinghu@northeastern.edu
**Scope**: Add a trailing, no-lookahead trend gate that drops non-trending names from the model score series so the deliverable top-N is no longer dominated by downtrends (空头排列 / "falling knives"). NO model retraining.

**TL;DR verdict**: The trend gate is a **live-usability filter, not an alpha/Calmar improver**. It takes the deliverable top-15 from **15/15 falling knives → 0/15**, but on the canonical backtest it **cuts net CAGR +19.05% → +5.3%, *raises* max drawdown (−56.7% → −64.9%), and crushes Calmar 0.34 → 0.08.** Root cause: the ensemble's short-term edge **is** a reversal/mean-reversion effect — the sub-MA20 names are exactly where the alpha lives, so filtering them removes the alpha. **Recommendation: do NOT wire the gate into the systematic strategy. Optionally expose the soft-gated list as a separate "可安心持有/discretionary" view with a disclaimer. The real Calmar levers are the P3 overlay (0.34→0.39, validated) + thicker alpha (P2b factors) + per-name stops — not trend-filtering.**

---

## 1. Problem

The production ensemble (LGBM + ALSTM) has a **reversal bias**: its highest-`score` names are disproportionately stocks in clear downtrends. Under the canonical `policy=fixed, top_k=5, period=5, profile=small, capital=100000` the long-window OOF (`production/reports/oof_2model_2021_2026.pkl`, 766 instruments, 2020-07-06 .. 2025-12-26, 1332 dates) earns +19.05% net CAGR / net_ir 0.67 — but the picks are "falling knives", which are (a) psychologically impossible for a human to hold and (b) drawdown-prone.

Goal (as posed): a configurable gate that removes names not in an acceptable trend, ideally improving **Calmar** and cutting falling-knife exposure at acceptable net-return cost.

## 2. Design

### 2.1 Where the gate sits
The backtest engine (`production/backtest/engine.py`) slices scores per date with `s.xs(d, level="datetime")` and hands that cross-section to the rebalance policy. So a gate that simply **removes `(datetime, instrument)` rows from the score Series** makes those names invisible to `_equal_top_k` / `Banded` — **no change to engine.py / rebalance.py**. Contract:
```python
production.trend_filter.apply_trend_filter(scores, prices_or_features, *, mode, drop_warmup=True) -> pd.Series
```
Output index is always a **subset** of input; surviving score values unchanged.

### 2.2 Trailing trend features (no-lookahead)
All features computed from closes **≤ D** only (decision after D's close, before next open). qlib forms in `TREND_FEATURE_EXPRS`: `$close`, `Mean($close,{5,10,20,60})`, `$close/Ref($close,20)-1` (20d momentum; positive lag = backward). Offline path uses `wide.rolling(w, min_periods=w).mean()` (trailing) + `wide/wide.shift(20)-1`. Boolean flags stored as nan-aware floats (fail=0.0 vs warm-up=NaN). No-lookahead is unit-tested (truncate-at-D invariance).

### 2.3 Gate modes
- **`none`** — passthrough (baseline).
- **`soft`** — keep iff `close > MA20`. Core falling-knife filter.
- **`strict`** — keep iff `MA5 > MA10 > MA20` AND `momentum_20 > 0` AND `close > MA20` (多头排列 + positive momentum). The extra `close>MA20` term guarantees **strict ⊆ soft**.
- Warm-up rows (incomplete window → NaN) are **dropped** by default (`drop_warmup=True`, conservative).

### 2.4 Orthogonal to the P3 overlay
The gate decides *which names are eligible* (cross-sectional); the P3 overlay (`compute_exposure`) decides *how much total capital is deployed* (time-series). They compose; the eval runs all 3 gates × {no overlay, +overlay(ma60/band0.10)}.

## 3. Files
| file | role |
|------|------|
| `production/trend_filter.py` | the gate: `apply_trend_filter`, `compute_trend_features`, `is_downtrend`, `TREND_FEATURE_EXPRS/NAMES` |
| `production/tests/test_trend_filter.py` | 15 offline TDD tests (no qlib) — all green |
| `production/_eval_trend.py` | standalone long-window OOF eval + live falling-knife sanity check |
| this file | results spec |

## 4. TDD — test contract (15 tests, all PASS)
Index-subset for every mode; `none` passthrough; constructed downtrend removed + uptrend kept (soft & strict); **no-lookahead** (strict verdict at mid-panel D identical on full vs truncated panel); MA trailing-not-centered; strict ⊆ soft; FLAT fails strict; warm-up dropped by default; raw-close-Series input path; empty→empty.
```
F:/Tools/Anaconda/envs/qlib/python.exe -m pytest production/tests/test_trend_filter.py -v   # 15 passed
```

## 5. Evaluation methodology
`production/_eval_trend.py` (run with `-m`, output → `logs/eval_trend.log`): load `oof_2model_2021_2026.pkl` → `score`; one `QlibDataLoader` call for trailing features (note: flatten its MultiIndex columns); `load_fwd_returns` once; for each gate × {no overlay, +overlay} call `build_report(..., policy_name="fixed", top_k=5, period=5, capital=100000, profile="small", exposure=...)`. **Policy is `fixed`/hold-5/5-day — the canonical config behind the +19% baseline (`2026-06-01-longwindow-lgbm-results.md`); `banded` underperforms it (`2026-05-31-net-return-results.md`).**

Reproduce:
```
F:/Tools/Anaconda/envs/qlib/python.exe -X utf8 -m production._eval_trend > logs/eval_trend.log 2>&1
```
(also writes `logs/eval_trend_summary.json`.) Sanity: the baseline row below reproduces the handover's +19.05%/0.67/−56.75%/Calmar0.34 and baseline+overlay reproduces the risk-overlay spec's +10.98%/−28.18%/0.39 — the eval is trustworthy.

## 6. Comparison table  (policy=fixed top_k=5 period=5, ¥100k, 小成本)

Gate retention: soft keeps **48.0%** of (date,inst) rows, strict **27.6%**; both retain all 1332 dates (no date-axis bias).

```
variant                       net_cagr  net_ir  turnover    max_dd  Calmar    win   days
-----------------------------------------------------------------------------------------
baseline (no gate)             +19.05%    0.67     0.173   -56.75%    0.34  50.45%  1332
soft (close>MA20)               +5.29%    0.32     0.183   -64.92%    0.08  48.65%  1332
strict (align+mom)              +5.51%    0.33     0.186   -60.81%    0.09  49.10%  1332
baseline (no gate) +overlay    +10.98%    0.57     0.110   -28.18%    0.39  46.25%  1332
soft (close>MA20) +overlay      +0.12%    0.12     0.115   -53.15%    0.00  44.89%  1332
strict (align+mom) +overlay     -0.53%    0.08     0.117   -40.84%   -0.01  45.12%  1332
```

Per-regime net_cagr (`metrics_net.DEFAULT_REGIMES`):
```
variant                       2020-02   2021-03   2023-01
---------------------------------------------------------
baseline (no gate)           +110.87%    +4.82%   +13.73%
soft (close>MA20)            +114.16%   +22.71%   -18.01%
strict (align+mom)            +58.41%    +5.37%    -3.37%
baseline (no gate) +overlay   +78.64%   +10.46%    +0.31%
soft (close>MA20) +overlay    +93.72%   +14.17%   -20.10%
strict (align+mom) +overlay   +30.91%    +8.84%   -11.42%
```

## 7. Interpretation — why the gate backfires

1. **The gate removes the alpha, not the risk.** net CAGR collapses +19.05% → +5.3% (soft/strict), and max drawdown gets **worse** (−56.7% → −64.9% soft). Calmar 0.34 → 0.08. With the overlay on top, the soft gate nets +0.12% (essentially zero).
2. **The model's short-term edge is a reversal/mean-reversion effect.** Its top picks sit below MA20 because that is precisely where its predicted bounce lives. Filtering "close>MA20" deletes the names the model is actually right about (in aggregate, as a diversified 5-name basket rebalanced every 5 days).
3. **Regime split confirms it's not purely a 2020 artifact.** The baseline is positive in all three regimes (+110.9% / +4.82% / +13.73%) — the 2020 COVID bounce is the biggest but 2021 & 2023 are also positive. The soft gate *helps* 2021 (+4.82%→+22.71%, a brief trend-following window) but *destroys* 2023 (+13.73%→−18.01%); net across regimes it loses badly. A blunt one-size trend filter is regime-wrong as often as it is regime-right.
4. **Turnover barely changes** (0.173→0.183), so the damage is selection quality, not cost.

## 8. Live falling-knife sanity check
Latest cross-section of `examples/mlruns/pred_2026-06-02.pkl` (2026-06-02, 842 instruments), top-15 by raw `score` vs after the soft gate:
- **RAW top-15: 15 of 15 are falling knives** (close<MA20), e.g. SH603737 (−22.3% vs MA20), SZ002335 (−32.8%), SH688281 (−30.9%) — every single highest-conviction pick is in a downtrend.
- **SOFT-GATED top-15: 0 of 15 are falling knives** (by construction). The gated names (SZ301236 +7.0%, SH600008 +4.2%, SH601088 +5.0%, large-cap state-owned banks/utilities …) are above MA20 — but they are far lower in the model's raw ranking (scores ~−180 to −250 vs the raw top's ~−40 to −140), i.e. names the model has *low* conviction in.

So the gate trades the model's high-conviction (downtrend) names for low-conviction (uptrend) names. Comfortable to hold; no demonstrated edge (§6).

## 9. Recommendation

1. **Do NOT add the trend gate to the systematic strategy ranking.** It is a net-negative on return, drawdown, and Calmar. (Clear negative result — recorded, not buried.)
2. **The model is best used as designed**: a *diversified, mechanically-rebalanced reversal basket* (hold 5 / 5-day, no discretion). For risk control, the validated lever is the **P3 exposure overlay** (Calmar 0.34→0.39, max_dd −56.7%→−28.2%) — keep using it. Next Calmar levers: **thicker alpha (P2b short-term factors, backfill running)** and **per-name stops / volatility sizing** — not cross-sectional trend filtering.
3. **Product option (discretionary view only):** expose the **soft-gated** list as an optional "可安心持有 / 顺势版" view on the Picks page for users who want to hand-pick a single name and cannot stomach a downtrend chart — with an explicit label that it **sacrifices the model's (reversal) edge** and is a comfort filter, not a higher-return list. Do not make it the default. This matches `_pick_trader.py`'s 推荐/回避 intent.
4. **Better "trend-aware" idea for the future (regime-conditional, not a post-filter):** feed trend state (close>MA20, MA-alignment, momentum) as **model features** so the model can *learn when reversal applies vs when trend applies* (the §7.3 regime flip suggests the right behavior is conditional, not constant). This is testable via the same factor-backfill machinery as P2b — a cleaner path than a blunt gate that uniformly deletes alpha.

## 10. Reusable artifacts
- `apply_trend_filter` / `compute_trend_features` / `is_downtrend` are general-purpose and unit-tested; the live soft gate is ready to drop into the Picks "discretionary" view (point 3) with no further modeling work.
- `logs/eval_trend.log` + `logs/eval_trend_summary.json` hold the raw numbers.
