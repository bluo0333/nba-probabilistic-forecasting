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
    expected_minutes: float | None = Field(default=None, ge=0, le=60)
    usage_adjustment_pct: float | None = Field(default=None, ge=-50, le=50)
    playoff_mode: bool = Field(default=False)
