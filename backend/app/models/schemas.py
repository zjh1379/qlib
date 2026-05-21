from pydantic import BaseModel, Field


class ScreenItem(BaseModel):
    rank: int                       # 1-based, by score_avg desc
    symbol: str
    name: str = ""
    score_today: float              # latest day's raw score
    score_avg: float                # mean score over `days` window
    rank_avg: float                 # mean cross-sectional rank over `days` window (lower = better)
    days_in_top: int                # how many of the `days` window days this symbol was in top-N


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
