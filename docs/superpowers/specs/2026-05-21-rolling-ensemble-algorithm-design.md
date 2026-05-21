# Rolling Multi-Model Ensemble · Algorithm Design Spec

**Date**: 2026-05-21
**Status**: Approved for implementation planning
**Owner**: zhu.jinghu@northeastern.edu
**Replaces**: the single-model `daily_cn_fresh` baseline (LightGBM-Alpha158, IC 0.024, IR 1.96 on a single 16-month backtest with survivorship bias)

---

## 1. Context

The current production model is a single LightGBM trained once on CSI300 / Alpha158 / 2018-2023 data, validated on 2024, and used unchanged to predict 2025+. Honest assessment of its weaknesses:

- **Single model** — no diversity; the IC=0.024 number itself is at the ceiling for CSI300 + Alpha158 + LightGBM.
- **No rolling retrain** — model is 1.5+ years stale; will drift further as market regime evolves.
- **Survivorship bias** — training universe is *today's* CSI300, so historical samples are filtered through "stocks that survived to be index members today".
- **Label = T+1 close → T+2 close** — requires order-at-close, which manual retail trader cannot execute.
- **No transaction cost / turnover penalty** — model may produce high-turnover predictions with negative real P&L after fees.
- **Rank-average ensemble** — not applicable (only one model), but the design hasn't anchored a better path.

P1.5 already shipped a working Picks page + data scope expansion. We now have data for 800+ instruments (CSI300 + CSI500 + 41 ETFs + custom). The algorithm needs to graduate from prototype to a production-grade quant pipeline.

## 2. Goals & Non-goals

### Goals (β phase)

1. Replace single LightGBM with a **3-model ensemble**: LightGBM + ALSTM + TRA.
2. Stand up a **weekly walk-forward rolling retrain** pipeline (default Sunday 22:00, **configurable in-app**).
3. Fix five quant-research-grade issues: PIT constituents, realistic label, multi-horizon, stacking, turnover smoothing.
4. Expand training universe to **CSI800** (CSI300 + CSI500 union, with PIT filtering).
5. Add evaluation discipline: multi-metric scorecard, regime split, shadow paper trading, auto rollback.
6. Integrate cleanly with existing Picks / Charts / Dashboard pages.
7. Reach **IC mean ≥ 0.030, IR ≥ 2.5 (with transaction cost), max drawdown ≤ 15%** on a multi-regime backtest.

### Non-goals (deferred to γ phase or beyond)

- MASTER (2024 Transformer SOTA) — added in γ phase after β validation.
- Industry / sector-neutral predictions.
- Style factor neutralization (Beta, Size, Value).
- Regime-conditional models (different model weights per regime).
- Custom fundamental factor library (PE / ROE / forecast EPS).
- Per-stock SHAP explainability.
- Intraday or minute-level predictions.
- Custom portfolio construction (we keep TopK=30, dropout=5 simple strategy).
- A/B framework beyond shadow paper trading.

### Out of scope (forever)

- Real broker integration (we stay decision-support only per P1 spec).
- Public-internet deployment (Tailscale-only).

## 3. Decisions Locked Through Brainstorming

| # | Decision | Choice | Rationale |
|---|---|---|---|
| Q1 | "Better" means | **D** (all three: model + robustness + multi-model consensus) | User wants real-world utility, not academic novelty |
| Q2 | Model menu | **β**: LightGBM + ALSTM + TRA | Best ROI given GPU available (3080Ti / 5070Ti) |
| Q3 | Rollout cadence | **B**: β first (2-3 weeks), then γ adds MASTER | De-risks via staged delivery; β is independently shippable |
| Q4 | Training universe | **CSI800** (CSI300 + CSI500) | Picks page already shows CSI500 stocks; ~2x training data; standard quant benchmark |
| Fix-1 | Universe bias | **PIT constituents** (monthly snapshots) | Eliminates survivorship inflation; critical for trustworthy backtests |
| Fix-2 | Label realism | **Open-to-open**: `Ref($open, -2) / Ref($open, -1) - 1` | Matches manual retail execution; user trades morning, not at close |
| Fix-3 | Forecast horizon | **Multi-horizon**: 1d + 5d + 20d in a single model (multi-task head) | Diversification; 5d label has 70% less noise than 1d |
| Fix-4 | Ensemble method | **Stacking** (Ridge meta-learner on 9 base preds, OOF training) | Auto-weights base models by recent performance |
| Fix-5 | Turnover control | **EWMA smoothing**: `score_t = 0.5 · score_t + 0.5 · score_t-1` | Cuts daily churn ~50%; minimal IC loss |

## 4. Architecture

```
┌──────────────────┐  ┌──────────────────┐
│   Data + PIT     │→ │  Alpha158        │ →┐ (flat features)
│  cn_data_bs/     │  │  Alpha360        │ →┤ (price-vol series)
│  + PIT 月度快照   │  └──────────────────┘  │
└──────────────────┘                        │
                                            ▼
                          ┌─────────────────────────────┐
                          │  Base Models (β phase)      │
                          │                             │
                          │  LightGBM-Alpha158          │
                          │    × 3 horizons (1d/5d/20d) │
                          │                             │
                          │  ALSTM-Alpha360             │
                          │    multi-head 3 horizons    │
                          │                             │
                          │  TRA-Alpha360               │
                          │    multi-head 3 horizons    │
                          │                             │
                          │  → 9 raw scores per stock   │
                          └─────────────────────────────┘
                                            │
                                            ▼
                          ┌─────────────────────────────┐
                          │  Stacking Meta-Learner      │
                          │  Ridge regression (α=1.0)   │
                          │  trained on OOF base preds  │
                          │  + cross-sectional z-score  │
                          └─────────────────────────────┘
                                            │
                                            ▼
                          ┌─────────────────────────────┐
                          │  Post-process               │
                          │  EWMA: 0.5 · t + 0.5 · t-1  │
                          └─────────────────────────────┘
                                            │
                                            ▼
                                    rolling_v2_ensemble
                                    pred.pkl + per-stock
                                    base scores + consensus
                                            │
                                            ▼
                          ┌─────────────────────────────┐
                          │  Picks / Charts / Dashboard │
                          └─────────────────────────────┘

  ┌────────────────────────────────────────────────────┐
  │  Weekly rolling retrain · configurable (default Sun 22:00) │
  │                                                    │
  │  |── 5y Train ──|── 1y Valid ──|── 1y Stack ──|── 1w Test ──|
  │                                                    │
  │  Per-horizon train length:                          │
  │   1d → 3y (high-freq decay)                         │
  │   5d → 5y (standard)                                │
  │   20d → 7y (long signal needs long sample)          │
  └────────────────────────────────────────────────────┘
```

### Stack components

| Layer | Technology |
|---|---|
| Backend | FastAPI (existing) |
| ML framework | qlib + PyTorch (via existing `F:\Tools\Anaconda\envs\qlib` env) |
| Tree model | LightGBM (CPU) |
| NN models | PyTorch on CUDA (3080Ti / 5070Ti) |
| Meta-learner | scikit-learn Ridge |
| Tracking | mlflow file store (existing `examples/mlruns/`) |
| Scheduling | APScheduler in FastAPI lifespan, config persisted in DB, UI-editable |
| Data source | baostock (PIT) + cn_data_bs (existing) |

## 5. Model Details

### Model A · LightGBM-Alpha158

| Field | Value |
|---|---|
| qlib class | `qlib.contrib.model.gbdt.LGBModel` |
| Feature handler | `qlib.contrib.data.handler.Alpha158` |
| Hyperparameters | `lr=0.05, max_depth=8, num_leaves=210, lambda_l1=205, lambda_l2=580, num_boost_round=1000, early_stopping_rounds=50` |
| Multi-horizon | 3 independent LGBModel instances (shared features, different labels) |
| Training time | CPU ~5 min × 3 horizons (parallelizable on multi-core) |
| Single-model IC target | 0.025-0.035 |

### Model B · ALSTM-Alpha360

| Field | Value |
|---|---|
| qlib class | `qlib.contrib.model.pytorch_alstm_ts.ALSTM` |
| Feature handler | `qlib.contrib.data.handler.Alpha360` (NOT Alpha158) |
| Sequence length | 20 trading days |
| Input dim | 6 (OHLC + volume + change) |
| Hyperparameters | `hidden_size=64, num_layers=2, dropout=0.0, batch_size=2048, n_epochs=100, lr=1e-3, early_stop_patience=20` |
| Multi-horizon | Single network, 3 output heads (multi-task loss = sum of 3 IC losses) |
| Training time | RTX 3080Ti ~25 min |
| Single-model IC target | 0.030-0.040 |

### Model C · TRA-Alpha360

| Field | Value |
|---|---|
| qlib class | `qlib.contrib.model.pytorch_tra.TRA` |
| Feature handler | `qlib.contrib.data.handler.Alpha360` |
| Sequence length | 20 trading days |
| Hyperparameters | `hidden_size=64, num_states=10, lamb=1.0, transport_method=optimal-transport, batch_size=1024, n_epochs=100, lr=1e-3` |
| Multi-horizon | Native multi-task (TRA design intent) |
| Training time | RTX 3080Ti ~50 min |
| Single-model IC target | 0.035-0.045 |

### Why both ALSTM and TRA

| Aspect | ALSTM | TRA |
|---|---|---|
| Strategy | Universal predictor (one model fits all stocks) | Mixture of K=10 specialized predictors |
| Routing | Same path for every stock | Optimal-transport router selects head per stock |
| Strength | Robust general patterns | Handles distributional shift & long-tail |
| Weakness | Slow to adapt to regime change | Can overfit when routing degenerates |

They make different errors → ensemble diversity gain.

### Training notes

- Train models **serially** to avoid GPU contention.
- Always train LightGBM first (fastest, guaranteed-success baseline).
- ALSTM/TRA failure → ensemble degrades to 2-model or 1-model rank-average fallback.
- Apply `RobustZScoreNorm` processor before NN training to clip 1% extremes.
- Apply gradient clipping `clip_grad_norm_(max_norm=3.0)` in NN loops.

## 6. Training Pipeline

### Stages (run order; offsets measured from trigger time T₀, default Sunday 22:00)

```
T₀+00:00  ① Data integrity check (re-run incremental refresh)
T₀+00:02  ② Build PIT training universe (CSI300 + CSI500 monthly snapshots)
T₀+00:05  ③ Train base models (sequentially):
          ├─ LightGBM × 3 horizons     (15 min, CPU)
          ├─ ALSTM multi-head          (25 min, GPU)
          └─ TRA multi-head            (50 min, GPU)
T₀+01:35  ④ Generate OOF base predictions on Stack-fit 1y window
T₀+01:40  ⑤ Fit stacking Ridge meta-learner (<1 min, CPU)
T₀+01:41  ⑥ Generate predictions for next week's test window
          └─ EWMA post-processing
T₀+01:45  ⑦ Backtest evaluation (8 metrics, multi-regime split)
T₀+01:50  ⑧ Compare vs last-week model (paired t-test) → decide swap or hold
T₀+01:55  ⑨ Write mlruns + invalidate /api/models cache
```

Total: ~2 hours wall clock.

### Schedule configuration (in-app)

The trigger time is **configurable inside the app** (no Windows Task Scheduler dependency):

- DB table `retrain_schedule` with single-row config: `day_of_week` (0=Mon … 6=Sun), `hour`, `minute`, `enabled`, `last_run_at`, `next_run_at`.
- Default value seeded on first boot: `(6, 22, 0, enabled=true)` — Sunday 22:00.
- Backend uses **APScheduler** running in the FastAPI process; it reads config at startup and on every PUT.
- On config change: drop the current job, recompute next-run, schedule new job. No restart needed.
- Trading-hours guard: if user picks a slot inside `09:30-15:00 CST` on a weekday, backend rejects with `400 trigger_during_trading_hours`.
- Manual "Run now" button calls `POST /api/scheduling/retrain/run-now` which spawns the same pipeline as the cron path (subject to the same trading-hours guard with a confirm-override flag).

### PIT (Point-in-Time) constituent handling

`production/pit_constituents.py` (new):

- Calls `bs.query_hs300_stocks(date=X)` and `bs.query_zz500_stocks(date=X)` for first-of-month dates over past 7 years.
- Caches to `production/pit_constituents.parquet`.
- For each training sample (stock, date), only include if stock was in CSI300 ∪ CSI500 on that month's first.
- Sanity check: each monthly snapshot must contain 280-320 CSI300 entries and 480-520 CSI500 entries; out-of-range → throw + use last successful cache.

### Walk-forward window per horizon

| Horizon | Train length | Valid length | Stack-fit length | Test length |
|---|---|---|---|---|
| 1d | 3 years | 1 year | 1 year | 1 week |
| 5d | 5 years | 1 year | 1 year | 1 week |
| 20d | 7 years | 1 year | 1 year | 1 week |

Each week, all windows slide forward by 7 days.

### Failure handling

| Failure point | Action |
|---|---|
| ① Data refresh fails | **Abort pipeline**, alert, keep last week's model |
| ② PIT fetch fails | Use last cached PIT (≤ 4 weeks stale acceptable) |
| ③ Single base model fails | Skip it; ensemble adapts (2 or 1 model fallback) |
| ③ All 3 base models fail | Abort, alert, keep last week's model |
| ⑤ Stacking fit fails | Fall back to equal-weight rank-average ensemble |
| ⑦ Backtest IR drops > 50% vs last week | **Don't swap**, alert, manual review required |

## 7. Ensemble & Stacking

### Why Ridge (vs alternatives)

- vs **OLS**: 9 base preds are correlated (0.5+ pairwise); Ridge stabilizes coefficients.
- vs **Lasso**: We don't want zero coefficients (we want soft downweighting, not "delete this model").
- vs **LightGBM-as-meta**: ~200k OOF samples × 9 dims is too few for GBM to avoid overfitting at the meta layer.

### Stacking input

Per stock per day, 9-dim vector:

```
[lgbm_1d, lgbm_5d, lgbm_20d, alstm_1d, alstm_5d, alstm_20d, tra_1d, tra_5d, tra_20d]
```

Each dimension is cross-sectionally z-scored per day before stacking:

```python
for col in base_pred_cols:
    df[col + '_z'] = df.groupby('date')[col].transform(lambda x: (x - x.mean()) / x.std())
```

### OOF training (critical detail)

- Train base models on `Train 5y + Valid 1y`.
- Apply trained models to `Stack-fit 1y` (data base models never saw) → OOF preds.
- Fit Ridge on (OOF preds, realized open-to-open returns).
- Test on `Test 1w`.

This prevents meta-learner from learning base models' in-sample leakage.

### Hyperparameter

- `Ridge(alpha=...)` with weekly grid search over `[0.1, 1.0, 10.0]`; pick max valid IC.
- **No other meta hyperparameters** (no polynomial features, no interaction terms).

### Consensus output

In addition to the ensemble score, compute:

```python
def consensus_score(base_preds_9d):
    signs = np.sign(base_preds_9d)
    return abs(signs.sum()) / 9.0   # range [0, 1]
```

- 9/9 same direction → 1.0 (very strong consensus)
- 5/9 same direction → 0.11 (basically noise)

Saved to `pred.pkl` alongside score for UI consumption.

### Fallback chain

```
Stacking (Ridge OOF)
  ↓ failure
Rank-average across available base models
  ↓ all base models failed
Roll back to last week's recorder
  ↓ no last week recorder (cold start)
Alert + halt swap; use baseline LGBModel
```

## 8. Evaluation & Backtest Protocol

### 8-metric scorecard

**Signal purity (model-level)**
- IC mean ≥ 0.030
- RIC mean ≥ 0.025
- ICIR ≥ 0.40
- Top-Bottom Spread ≥ 1.5% / month

**Portfolio performance (TopK=30, dropout=5)**
- Annualized excess return (cost-adjusted) ≥ +15%
- Information Ratio (cost-adjusted) ≥ 2.0
- Max drawdown ≤ 15%

**Reality check**
- Daily turnover ≤ 20%
- Single-stock max weight ≤ 10%

**Any single metric failing → don't deploy.**

### Multi-regime split

Run backtest separately on:

```
2018-01 to 2018-12  | Bear + trade war
2019-01 to 2020-02  | Recovery + bull start
2020-03 to 2021-02  | COVID + liquidity
2021-03 to 2022-10  | High volatility + Fed hikes
2022-11 to 2024-12  | AI rally + CSI300 underperformance
```

**Every segment's IR must be > 0**. If any single segment has negative IR, model is not regime-robust.

This is an offline, one-time test (re-run only on major architecture changes).

### Significance test (vs last week)

```python
from scipy.stats import ttest_rel
t_stat, p_value = ttest_rel(new_daily_ic, old_daily_ic)
```

Decision rule:
- `p < 0.05` AND `mean_diff ≥ 0.005` → swap
- else → keep last week's model

### Shadow paper trading

After every new model trains:
1. Mark new model as `shadow_v2_ensemble`.
2. For 4 weeks, both old (`rolling_v2_ensemble`) and shadow run predictions.
3. Frontend shows old as the official Picks; shadow runs invisibly.
4. After 4 weeks: compare shadow paper P&L vs production P&L.
5. If shadow IR > production IR + 0.5 → swap.
6. Else: discard shadow, keep training new candidates.

### Auto rollback

Continuously monitor:

```
if past 2 weeks cumulative IR < 0:
    auto rollback to N-2 recorder
if single week drawdown > 8%:
    halt + alert + human takeover
```

Rollback = pointer change in mlruns; immediate effect on `/api/models/screen`.

### Monthly review report

`production/reports/<year>_<month>.md` auto-generated on 1st of each month:
- Per-week IR table
- Base model contribution breakdown
- Stacking coefficient evolution
- Top-30 hit rate analysis
- 5 hand-selected "high-score-but-fell" failures for debugging

## 9. App Integration

### Backend API extensions

```
GET  /api/models/screen                       (existing — auto consumes new ensemble)
GET  /api/models/screen?view=ensemble         (new — defaults to this)
GET  /api/models/screen?view=lightgbm         (new — single base model view)
GET  /api/models/screen?view=alstm
GET  /api/models/screen?view=tra
GET  /api/models/shadow                       (new — shadow comparison)
GET  /api/models/version                      (new — current/last/last-2 recorder metadata)
POST /api/models/rollback                     (new — manual rollback to N-1 recorder)

GET  /api/scheduling/retrain                  (new — read schedule config)
PUT  /api/scheduling/retrain                  (new — update schedule config)
POST /api/scheduling/retrain/run-now          (new — manual trigger; honors trading-hours guard)
```

### Schema additions

`ScreenItem` (existing, extended):

```python
class ScreenItem(BaseModel):
    rank: int
    symbol: str
    name: str
    score_today: float
    score_avg: float
    rank_avg: float
    days_in_top: int
    # === added in this spec ===
    consensus: float = 0.0
    base_scores: dict[str, float] = {}   # e.g. {"lgbm_5d": 0.03, "alstm_5d": 0.02, ...}
```

### Frontend changes

**Picks page** (~1.5 days):
- View selector dropdown (Ensemble / LightGBM / ALSTM / TRA).
- Consensus column with color coding (≥ 7/9 green, 4-6 yellow, < 4 gray).
- Consensus filter slider on the filter bar.

**Charts page** (~1.5 days):
- Sub-view selector for per-model prediction overlay.
- Optional consensus band overlay on main chart.

**Dashboard** (~1 day):
- Model version card (current recorder, last-week IR comparison, next retrain countdown).
- Shadow monitor card (production vs shadow paper P&L during the 4-week shadow window).
- Base contribution bar chart (Ridge coefficients).

**Settings page** (~0.5 day):
- Retrain schedule editor: day-of-week selector + time picker + enabled toggle.
- Shows current `next_run_at` and `last_run_at`.
- "Run now" button (disabled during trading hours unless force-confirmed).

**Backend** (~1 day) — new endpoints listed above.

Total frontend effort: **~5 days** as P3.5 patch, parallel to or after β-phase ML work.

## 10. Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| R1 NN training instability (NaN, divergence) | 30% | High | Clip features (`RobustZScoreNorm`); gradient clip (max_norm=3); lr auto-fallback to 5e-4 |
| R2 PIT data gaps from baostock | 40% | High | Sanity range check; fallback to last cached PIT; akshare as secondary source |
| R3 Time budget overrun (3w → 5-6w) | 70% | Medium | Milestone-based delivery (Week 1: LightGBM-only swap; Week 2: 2-model; Week 3: full 3-model) |
| R4 GPU OOM | 15% | Low | Serial training; batch_size auto-halve on OOM |
| R5 mlruns directory bloat | 100% (time) | Low | Auto-archive recorders > 8 weeks old to `production/archive/` |
| R6 Mid-day model swap during trading | Low | High | "Retrain now" disabled 09:30-15:00; confirm dialog |
| R7 Backtest overfitting (silent killer) | 60% | Critical | Lock hyperparameters in spec; tune only on major architecture changes; **shadow paper P&L is the ground truth** |
| R8 Walk-forward window off-by-one | 50% | Critical | Mandatory unit test `test_no_overlap_train_valid_test`; fixed-data integration tests |
| R9 Survivorship bias leakage via factor calc | 30% | Critical | Factor computation universe = PIT all-A, not just CSI800; regression test with known delisted stock |
| R10 Stacking meta over-fit | 35% | Medium | Lock Ridge inputs to 9 base z-scored preds (no interaction terms); grid search α over [0.1, 1.0, 10.0] |

### Early warning signals

- R1: Week-1 ALSTM IC < 0.020
- R2: log contains `pit_constituents_undersized` warning
- R3: Week-2 milestone not reached on ALSTM
- R5: `du -sh examples/mlruns/` > 5GB
- R7: shadow IR < 60% of backtest IR
- R8: any test_no_overlap test failure
- R9: backtest regime-segment IR strongest on segments that should be hardest (e.g. 2018 bear, 2020 COVID)
- R10: Ridge R² on train > 5× R² on validation

## 11. Acceptance Criteria

β phase is "done" when:

**Functional**

- [ ] `production/rolling_train.py` runs end-to-end without manual intervention
- [ ] Weekly APScheduler job triggers at user-configured time (default Sunday 22:00) and completes within 2 hours
- [ ] `/api/scheduling/retrain` GET/PUT works and rejects trading-hours slots
- [ ] `pred.pkl` from latest ensemble recorder is consumed by Picks / Charts pages without code changes
- [ ] All 3 base models trained; ALSTM and TRA use GPU; LightGBM uses CPU
- [ ] PIT constituents file regenerated monthly; passes sanity range check
- [ ] Ensemble output includes per-stock `consensus` field (0-1)
- [ ] Shadow paper trading framework runs new candidates in parallel for 4 weeks before swap
- [ ] Auto-rollback to N-2 recorder triggers on 2-week negative IR

**Performance** (measured on PIT-corrected 2024-2025 backtest)

- [ ] IC mean ≥ 0.030
- [ ] IR (cost-adjusted) ≥ 2.5
- [ ] Max drawdown ≤ 15%
- [ ] Daily turnover ≤ 20%
- [ ] All 5 regime-split segments have IR > 0

**Quality**

- [ ] Hyperparameters locked in YAML; no in-code tuning during weekly runs
- [ ] Unit tests for `test_no_overlap_train_valid_test`, PIT range check, stacking input dimensionality
- [ ] mlruns recorder versions for past 8 weeks retrievable
- [ ] All 8 evaluation metrics in mlruns metrics dict per recorder

## 12. γ Phase Plan (Future)

After β meets all acceptance criteria for **at least 4 consecutive shadow weeks**:

1. Add **MASTER (2024 Market-Aware Stock Transformer)** as 4th base model.
2. Ensemble grows from 9 → 12 base preds (4 models × 3 horizons).
3. Shadow trial 4 weeks → swap if IR diff > +0.5 vs β-3-model.
4. Optional: experiment with industry-neutral predictions in γ+.

γ is **not** part of this spec; tracked separately when triggered.

## 13. Glossary

| Term | Meaning |
|---|---|
| **PIT** | Point-in-Time. Using *that day's* universe membership, not today's, during historical training. |
| **Walk-forward** | Sliding-window validation: train on [T-N years, T], validate on [T, T+1y], test on [T+1y, T+1y+1w]. Slide weekly. |
| **OOF** | Out-of-Fold. Base model predictions on data the base model never saw during training; used to train the stacking meta-learner. |
| **Stacking** | Ensemble method where a meta-model is fit on base models' predictions instead of simple averaging. |
| **Consensus** | Fraction of base predictions agreeing in direction (sign). Range [0, 1]. |
| **EWMA** | Exponentially Weighted Moving Average. Used here as `score_t = α·score_t + (1−α)·score_{t−1}` with α=0.5 for turnover smoothing. |
| **Ridge** | L2-regularized linear regression. Used as the meta-learner in stacking. |
| **IC / RIC** | Information Coefficient / Rank Information Coefficient. Daily cross-sectional Pearson/Spearman correlation between predicted and realized returns. |
| **IR** | Information Ratio. Annualized excess return ÷ tracking error. Risk-adjusted alpha. |
| **TopK Drop** | Portfolio construction: hold top K predicted stocks, drop the lowest M each rebalance. |
| **ALSTM** | Attention-LSTM. RNN with attention over hidden states. |
| **TRA** | Temporal Routing Adapter. Multi-head RNN with optimal-transport routing per stock per day. |
| **MASTER** | Market-Aware Stock Transformer (2024). Transformer with explicit market-state embedding. γ-phase candidate. |
| **Shadow paper trading** | Running a candidate model in parallel with production, comparing assumed P&L without affecting production output, for N weeks before deciding to swap. |
| **Regime split** | Backtesting separately on multiple historical periods chosen to represent different market environments. |

---

**Next step**: After user review, invoke `superpowers:writing-plans` to produce a step-by-step implementation plan for the β phase (LightGBM multi-horizon → ALSTM → TRA → stacking → evaluation → integration).
