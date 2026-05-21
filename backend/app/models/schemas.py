from pydantic import BaseModel, Field


class ScreenItem(BaseModel):
    rank: int                       # 1-based, by score_avg desc
    symbol: str
    name: str = ""
    score_today: float              # latest day's raw score
    score_avg: float                # mean score over `days` window
    rank_avg: float                 # mean cross-sectional rank over `days` window (lower = better)
    days_in_top: int                # how many of the `days` window days this symbol was in top-N
    consensus: float = 0.0
    base_scores: dict[str, float] = Field(default_factory=dict)


class ScreenResponse(BaseModel):
    experiment: str
    recorder_id: str
    latest_date: str                # ISO date of last prediction day
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
