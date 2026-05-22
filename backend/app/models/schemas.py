from pydantic import BaseModel, Field


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


class ScreenResponse(BaseModel):
    experiment: str
    recorder_id: str
    latest_date: str                # ISO date of last prediction day
    window_days: int
    universe_size: int
    items: list[ScreenItem]


class CandidatesResponse(BaseModel):
    """Same shape as ScreenResponse; semantically a 'no filters applied' candidate pool
    intended for client-side filter + sort. Returned by GET /api/models/candidates."""
    experiment: str
    recorder_id: str
    latest_date: str
    window_days: int
    universe_size: int
    items: list[ScreenItem]


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
