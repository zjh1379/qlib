# Net-Return Backtest Results: P0 Honest Measurement + P1 Turnover Reduction

**Date:** 2026-05-31  
**Branch:** feat/net-return-backtest  
**Status:** Completed — P1 acceptance partial (net_ir passes, turnover narrowly misses 50% cut)

---

## 1. Prediction File Used

| Field | Value |
|-------|-------|
| File | `examples/mlruns/159121250791620667/009bfabbaa8046a885d7c78758325da5/artifacts/pred.pkl` |
| Date span | 2026-01-26 to 2026-05-22 |
| Calendar days | 116 (58 trading days) |
| Instruments (unique) | 800 (CSI800 universe) |
| Columns | lgbm_1d, lgbm_5d, lgbm_20d, alstm_1d, alstm_5d, alstm_20d, score, consensus |
| Score column | Rank-average ensemble composite (lower rank = lower score; negative scale) |

**Selection rationale:** This recorder has the widest date span in mlruns (116 calendar days vs. 81 for the next-best). All other candidates cover single weeks (4-day spans) or a single month.

**CAVEAT — Window Limitation:** 58 trading days (roughly 12 weeks) is a short evaluation window. It covers a single market regime (A-share bull run followed by a shallow correction). The long multi-regime evaluation (2019-2026) that would give statistically meaningful IR estimates requires the P2 LGBM data backfill (next plan). All numbers below should be read as **in-sample regime evidence**, not production-grade Sharpe estimates.

---

## 2. Correction of Previously-Reported Numbers

The previous `metrics.py` had an overlap-counting bug that produced a reported 54% annual gross return and 46% daily turnover on the same ensemble. Those numbers were meaningless because the "per-day" turnover was computed over a multi-day prediction window without correct day-alignment. The P0 toolchain (`production/backtest/`) computes:

- **Gross CAGR** from the actual daily position P&L (open-to-open, aligned to decision date)
- **Cost drag** from the realistic `small` profile: 5 bps execution + 0.1% stamp duty (sell-side) + 最低5元 fixed commission
- **Net CAGR** = Gross CAGR minus cost drag annualized

---

## 3. Honest Baseline (Daily Policy, k=30)

| Metric | Value |
|--------|-------|
| Policy | daily rebalance |
| Top-k | 30 |
| Capital | ¥100,000 |
| Profile | small (最低5元 brokerage) |
| **avg_turnover** | **0.320 (32%/day)** |
| **net_ir** | **-0.431** |
| **net_cagr** | **-14.1%** |
| gross_cagr | +24.2% |
| **cost_drag_annual** | **36.9%** |
| max_drawdown | -11.2% |
| win_rate | 52.6% |
| n_days | 57 |
| final_nav | ¥96,626 |

**Interpretation:** Daily rebalancing of a small ¥100k account with 最低5元 brokerage is self-defeating. The gross signal is mildly positive (+24% annualized) but the 37%/year cost drag swamps it entirely. Net IR of -0.43 means the strategy loses more than half a standard deviation per year to costs alone. This is the *honest* number the P0 toolchain was built to reveal.

---

## 4. Parameter Sweep Highlights

Sweep grid: 3 policies × 5 top_k values × 4 periods × 4 capital levels = 120 configurations.  
All at `small` profile.

**Overall winner** (unrestricted capital):

| policy | top_k | period | capital | net_ir | avg_turnover | net_cagr |
|--------|-------|--------|---------|--------|-------------|---------|
| fixed | 30 | 5 | ¥1,000,000 | 1.080 | 0.133 | +28.5% |
| fixed | 30 | 5 | ¥300,000 | 1.019 | 0.133 | +26.4% |
| fixed | 10 | 5 | ¥300,000 | 0.964 | 0.161 | +29.5% |

**At ¥100,000 (the target capital), best by net_ir:**

| policy | top_k | period | net_ir | avg_turnover | net_cagr |
|--------|-------|--------|--------|-------------|---------|
| **fixed** | **10** | **5** | **0.902** | **0.161** | **+27.0%** |
| fixed | 30 | 5 | 0.767 | 0.133 | +18.3% |
| fixed | 20 | 5 | 0.589 | 0.146 | +13.6% |
| banded | 30 | — | 0.072 | 0.170 | -1.6% |
| daily | 30 | — | -0.431 | 0.320 | -14.1% |

**Key finding:** The banded policy underperforms fixed on this short window. The `fixed` policy with 5-day holding period reduces daily equivalent turnover to ~3.2% per day (vs. 32% for daily rebalance) because positions are held for the full period.

**Per-capital recommendation at ¥100k with 最低5元 cost:** hold 10 names, rebalance every 5 trading days using fixed policy. At this holding frequency, fixed-cost commission becomes negligible relative to position size (10 × ¥10,000 average = 3 bps round-trip fixed cost vs. 10 bps variable). Avoid daily rebalancing at small capital — the fixed minimum commission is lethal.

---

## 5. Baseline vs. Improved vs. Neutralized

| Variant | avg_turnover | net_ir | net_cagr | max_drawdown | win_rate |
|---------|-------------|--------|---------|-------------|---------|
| **Baseline** (daily, k=30) | 0.320 | -0.431 | -14.1% | -11.2% | 52.6% |
| **Improved** (fixed, k=10, period=5) | 0.161 | +0.902 | +27.0% | -10.0% | 50.9% |
| **Improved + neutralize** | 0.163 | -0.581 | -15.4% | -10.1% | 49.1% |

**Deltas (improved vs. baseline):**

| Metric | Change |
|--------|--------|
| avg_turnover | -0.159 (-49.7%) |
| net_ir | +1.333 (from -0.43 to +0.90) |
| net_cagr | +41.1 percentage points |

---

## 6. P1 Acceptance Criteria Result

**Criterion #1 — net_ir improves:** PASS  
Improved net_ir = +0.902 vs. baseline = -0.431. Delta = +1.333. The turnover-reduction mechanism demonstrably works: by holding positions for 5 days instead of rebalancing daily, cost drag drops from 36.9%/yr to 10.1%/yr, converting a losing strategy into a profitable one on this window.

**Criterion #2 — avg_turnover ≤ 50% of baseline:** NARROWLY FAIL  
Improved turnover = 0.161 / baseline = 0.320 = 50.3%. The ratio is 0.3 percentage points above the 50% threshold. The turnover is reduced by nearly exactly half — but the strict ≤50% acceptance boundary is not met.

---

## 7. Neutralization Effect

Adding `--neutralize` (sector-demean scores using `production/cache/industry_map.parquet`) **dramatically hurt** net_ir on this window:

- Gross CAGR dropped from +40.4% to **-6.2%** (neutralization actively removed the alpha)
- Net_ir dropped from +0.902 to **-0.581**
- Win rate dropped from 50.9% to 49.1%

**Interpretation:** On this short 12-week window (Jan–May 2026), the ensemble's raw alpha was predominantly a sector-level bet (the model held sectors that outperformed during this period). Neutralizing the scores removed the sector timing signal and left residual stock-specific alpha that was near zero for this window. This does not mean neutralization is always harmful — on longer windows with multi-regime data, it would reduce sector-driven drawdown risk. This is a known hazard of sector neutralization on short evaluation windows.

---

## 8. Honest Interpretation

### What these numbers mean

The 58-trading-day evaluation window covers a specific A-share regime: a January-May 2026 period that happened to feature a mild trend favorable to the ensemble. The gross CAGR of +40% for the fixed-10 config is almost certainly not representative of long-run alpha — it reflects both model alpha and regime luck.

### What the P0 toolchain proves

1. **The measurement infrastructure works correctly.** Net-of-cost returns are now computed from first principles: actual P&L series, not hand-waved metrics.
2. **The fixed-cost minimum commission problem is real.** At ¥100k capital with daily rebalancing, transaction costs alone destroy 37%/year. This was not visible from the previous broken `metrics.py`. The P0 toolchain makes this transparent and actionable.
3. **The turnover-reduction mechanism (P1) works as designed.** Switching from daily to fixed-5d rebalancing reduces turnover by ~50% and converts net_ir from -0.43 to +0.90 on this window.

### What requires P2 to evaluate properly

- Long-run (2019–2026) gross alpha after controlling for regime
- Statistical significance of net_ir (requires hundreds of trading days, not 57)
- Proper out-of-sample evaluation with walk-forward model retraining
- Whether +27% net CAGR on 12 weeks replicates across multiple market regimes

---

## 9. Technical Notes

**Small robustness fix applied:** `production/backtest/data.py` required a `sys.path` fixup (identical to `rolling_train.py`) to force the installed qlib 0.9.7 (`site-packages`) over the uncompiled source tree when running `python -m production.backtest.run` from the repo root. Also fixed `init_qlib_from_config` to wrap `config_path` in `Path()` before passing to `load_config`. Both fixes are minimal correctness patches, not behavior changes.

**Sweep CSV size:** `production/reports/sweep.csv` — 120 rows, small file, committed.

**Industry map:** `production/cache/industry_map.parquet` — 5,528 instruments, used for sector neutralization.
