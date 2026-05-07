from __future__ import annotations

from pydantic import BaseModel


class MatchupPredictResponse(BaseModel):
    matchup: str
    home_team: str
    away_team: str
    home_win_probability: float
    away_win_probability: float
    predicted_winner: str


class PlayerPropPredictResponse(BaseModel):
    player: str
    line_type: str
    line_type_label: str
    side: str
    line: float
    odds: float
    predicted_mean: float
    std_dev: float
    hit_probability: float
    implied_probability: float
    edge: float
    mean_source: str
    base_predicted_mean: float | None = None
    context_adjustment: float | None = None
    expected_minutes: float | None = None
    baseline_minutes: float | None = None
    usage_adjustment_pct: float | None = None
    playoff_mode: bool = False


class PlayerRecentGameResponse(BaseModel):
    date: str
    opponent: str
    value: float
    mins: float
