from __future__ import annotations

from pydantic import BaseModel, Field


class MatchupPredictRequest(BaseModel):
    home: str = Field(min_length=1)
    away: str = Field(min_length=1)
    game_date: str | None = Field(default=None)


class PlayerPropPredictRequest(BaseModel):
    player: str = Field(min_length=1)
    line_type: str = Field(min_length=1)
    side: str = Field(min_length=1)
    line: float
    odds: float
