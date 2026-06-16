from pydantic import BaseModel, Field

from app.analysis.schemas import AiAnalysis


class HorizonPrediction(BaseModel):
    """Per-(stock, horizon) prediction packet.

    target_date is the trading day this horizon predicts (latest_date +
    1/5/20 trading days). pred_return is the calibrated expected return as
    a decimal (e.g. 0.032 = +3.2%), or None when no calibration is loaded.
    percentile is 0..100 with 100 = best. model_agreement is 0..1 fraction
    of the 3 models that agree on direction (NaN-skip).
    """
    target_date: str
    pred_return: float | None = None
    percentile: float
    model_agreement: float | None = None
    raw_scores: dict[str, float] = Field(default_factory=dict)


class ScreenItem(BaseModel):
    rank: int
    symbol: str
    name: str = ""
    score_today: float
    score_avg: float
    rank_avg: float
    days_in_top: int
    consensus: float = 0.0
    base_scores: dict[str, float] = Field(default_factory=dict)
    last_price: float | None = None
    # Per-horizon predictions (T9): {"1d": HP, "5d": HP, "20d": HP}
    horizons: dict[str, HorizonPrediction] = Field(default_factory=dict)

    # Tier 1 screener metrics — exposed to UI for at-a-glance display
    # AND for client-side filtering/sorting (no backend re-fetch on filter change).
    pct_change_1d: float | None = None
    pct_change_3d: float | None = None
    pct_change_5d: float | None = None
    pct_change_10d: float | None = None
    pct_change_20d: float | None = None
    amplitude: float | None = None
    vol_ratio: float | None = None
    is_new_high_20d: bool = False
    is_new_high_60d: bool = False
    is_new_high_120d: bool = False
    board: str | None = None  # main | gem | star | bj | etf | other
    is_st: bool = False

    # 客户端即时层用：每股最近 K 个交易日的逐日排名/分数（与
    # CandidatesResponse.window_dates 升序对齐；缺数据为 None）。
    daily_ranks: list[int | None] = Field(default_factory=list)
    daily_scores: list[float | None] = Field(default_factory=list)

    # AI 分析层 (解读 + 风险旗标) — attached at serving time, None when not yet generated
    ai_analysis: AiAnalysis | None = None


class ScreenResponse(BaseModel):
    experiment: str
    recorder_id: str
    latest_date: str                # ISO date of last prediction day
    window_days: int
    universe_size: int
    items: list[ScreenItem]


class CandidatesResponse(BaseModel):
    """Same shape as ScreenResponse; semantically a 'no filters applied' candidate pool
    intended for client-side filter + sort. Returned by GET /api/models/candidates.

    Includes the list of base model+horizon columns available in the underlying
    pred.pkl (e.g. lgbm_1d, lgbm_5d, tra_5d…) so the frontend can render a
    model picker, and the subset actually used for the current score when a
    custom `models` query param was supplied (None means the pool-time default).
    """
    experiment: str
    recorder_id: str
    latest_date: str
    window_days: int
    universe_size: int
    available_models: list[str] = Field(default_factory=list)
    active_models: list[str] | None = None
    items: list[ScreenItem]
    # 客户端即时持续性/窗口过滤用：最近 K 个交易日（升序 ISO 日期）。
    window_dates: list[str] = Field(default_factory=list)
    # Multi-horizon UX (T9): which date the predictions are AS-OF, what the
    # latest qlib bin data is, and how stale predictions are vs that data.
    as_of_date: str | None = None
    data_latest_date: str | None = None
    data_stale_days: int = 0


class PredictionPoint(BaseModel):
    date: str                       # ISO YYYY-MM-DD
    score: float
    rank: int                       # cross-sectional rank that day (1-based)
    universe_size: int
    base_scores: dict[str, float] = Field(default_factory=dict)


class PredictionHistory(BaseModel):
    symbol: str
    name: str = ""
    experiment: str
    points: list[PredictionPoint]


class ExperimentInfo(BaseModel):
    name: str
    latest_recorder_id: str
    latest_metrics: dict[str, float] = Field(default_factory=dict)   # IC, RIC, etc.


class ExperimentsResponse(BaseModel):
    experiments: list[ExperimentInfo]


class RecorderVersion(BaseModel):
    recorder_id: str
    experiment: str
    created_at: str            # ISO timestamp
    metrics: dict[str, float] = Field(default_factory=dict)


class VersionResponse(BaseModel):
    current: RecorderVersion
    previous: RecorderVersion | None = None
    previous_2: RecorderVersion | None = None
    next_retrain_at: str | None = None


class RollbackRequest(BaseModel):
    target: str = Field(
        default="previous_1",
        pattern="^(previous_1|previous_2)$",
        description="Which recorder to roll back to (previous_1 = the run before the current one).",
    )


class RollbackResponse(BaseModel):
    status: str           # "rolled_back" | "no_op"
    archived_recorder_id: str | None = None
    new_current_recorder_id: str | None = None
    reason: str | None = None


class RecomputeProgress(BaseModel):
    phase: str  # "load" | "score" | "metrics" | "enrich" | "done"
    percent: int  # 0..100 overall
    message: str = ""


class RecomputeJob(BaseModel):
    job_id: str
    status: str  # "running" | "done" | "failed"
    started_at: str
    finished_at: str | None = None
    error: str | None = None
    view: str = "ensemble"
    models: list[str] = Field(default_factory=list)
    progress: RecomputeProgress | None = None


class RecomputeRequest(BaseModel):
    view: str = "ensemble"
    models: list[str] = Field(default_factory=list)


class RecomputeTriggerResponse(BaseModel):
    status: str  # "started" | "already_running"
    job_id: str | None = None
