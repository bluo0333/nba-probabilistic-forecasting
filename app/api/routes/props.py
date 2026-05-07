from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.schemas.request import PlayerPropPredictRequest
from app.schemas.response import PlayerPropPredictResponse, PlayerRecentGameResponse
from app.services import model_service

router = APIRouter(prefix="/props", tags=["props"])
logger = logging.getLogger(__name__)


@router.get("/players", response_model=list[str])
def list_players() -> list[str]:
    try:
        return model_service.get_player_names()
    except Exception as exc:
        logger.exception("Failed to list players")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load players: {exc}",
        ) from exc


@router.get("/recent-games", response_model=list[PlayerRecentGameResponse])
def list_recent_games(
    player: str,
    line_type: str = "points",
    limit: int = 10,
) -> list[PlayerRecentGameResponse]:
    try:
        return [
            PlayerRecentGameResponse(**row)
            for row in model_service.get_player_recent_games(
                player=player,
                prop_type=line_type,
                limit=limit,
            )
        ]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to list recent player games")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load recent games: {exc}",
        ) from exc


@router.post("/predict", response_model=PlayerPropPredictResponse)
def predict_player_prop(payload: PlayerPropPredictRequest) -> PlayerPropPredictResponse:
    try:
        logger.info(
            "player_prop_prediction_request player=%s line_type=%s side=%s",
            payload.player,
            payload.line_type,
            payload.side,
        )
        result = model_service.predict_player_prop(
            player=payload.player,
            prop_type=payload.line_type,
            side=payload.side,
            line=payload.line,
            odds=payload.odds,
            expected_minutes=payload.expected_minutes,
            usage_adjustment_pct=payload.usage_adjustment_pct,
            playoff_mode=payload.playoff_mode,
        )
        return PlayerPropPredictResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to score player prop")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to score player prop: {exc}",
        ) from exc
