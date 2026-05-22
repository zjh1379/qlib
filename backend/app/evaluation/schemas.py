from __future__ import annotations

from pydantic import BaseModel, Field


class ScorecardData(BaseModel):
    """The 8-metric scorecard (per spec §8).

    All metrics computed against open-to-open returns + TopK=30 long-only
    portfolio with `bps` transaction cost adjustment.
    """
    ic_mean: float
    ric_mean: float
    icir: float
    top_bottom_spread_monthly: float    # in percent (e.g. 1.8 = 1.8%/month)
    annual_excess_return: float         # decimal (0.15 = +15%)
    ir: float
    max_drawdown: float                 # negative number (-0.12 = -12%)
    daily_turnover: float               # decimal (0.18 = 18%)


class AcceptanceResult(BaseModel):
    """Per-criterion pass/fail against spec §11 thresholds."""
    passed: bool
    details: dict[str, bool]


class RegimeMetrics(BaseModel):
    """One regime segment's scorecard + label (e.g. '2020-COVID')."""
    label: str
    start: str                          # ISO date
    end: str                            # ISO date
    sample_size: int                    # number of (date, symbol) pairs evaluated
    scorecard: ScorecardData


class RecorderSummary(BaseModel):
    """Lightweight summary for the list view. Computed without full eval."""
    recorder_id: str
    experiment: str
    run_name: str
    created_at: str                     # ISO timestamp
    pred_start: str | None = None       # earliest prediction date
    pred_end: str | None = None         # latest prediction date
    pred_rows: int | None = None        # total rows in pred.pkl
    has_eval: bool = False              # True if cache has a result for this recorder
    # Lightweight 'quick look' metrics when has_eval=True. Detail view fetches the full scorecard.
    ic_mean: float | None = None
    ir: float | None = None
    acceptance_passed: bool | None = None


class EvalResult(BaseModel):
    """Full evaluation result for one recorder."""
    recorder_id: str
    experiment: str
    run_name: str
    computed_at: str                    # ISO timestamp
    window_start: str                   # ISO date of earliest evaluated prediction
    window_end: str                     # ISO date of latest evaluated prediction
    sample_size: int                    # rows after label join
    top_k: int                          # portfolio TopK used
    cost_bps: float                     # cost adjustment used
    scorecard: ScorecardData
    regimes: list[RegimeMetrics]
    acceptance: AcceptanceResult


class CompareResult(BaseModel):
    """Side-by-side comparison of two recorders + paired t-test on daily IC."""
    a: EvalResult
    b: EvalResult
    paired_t_stat: float
    paired_p_value: float
    significant_at_05: bool             # True iff p < 0.05
    ic_delta: float                     # b.scorecard.ic_mean - a.scorecard.ic_mean
    ir_delta: float                     # b.scorecard.ir - a.scorecard.ir
    verdict: str                        # 'b significantly better', 'a significantly better', 'no significant difference'


class EvalRunRequest(BaseModel):
    recorder_id: str
    top_k: int = Field(default=30, ge=1, le=300)
    cost_bps: float = Field(default=10, ge=0)
    force_refresh: bool = False         # if True, bypass cache and recompute
