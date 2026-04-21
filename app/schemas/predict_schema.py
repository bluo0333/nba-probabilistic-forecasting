from __future__ import annotations

from pydantic import BaseModel, Field


class MatchupPredictRequest(BaseModel):
    home: str = Field(min_length=1)
    away: str = Field(min_length=1)
    game_date: str | None = Field(default=None)


class MatchupPredictResponse(BaseModel):
    matchup: str
    home_team: str
    away_team: str
    home_win_probability: float
    away_win_probability: float
    predicted_winner: str


class PlayerPropPredictRequest(BaseModel):
    player: str = Field(min_length=1)
    line_type: str = Field(min_length=1)
    side: str = Field(min_length=1)
    line: float
    odds: float


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
