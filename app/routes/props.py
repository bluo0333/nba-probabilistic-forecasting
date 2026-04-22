from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.predict_schema import (
    PlayerPropPredictRequest,
    PlayerPropPredictResponse,
)
from app.services import model_service

router = APIRouter(prefix="/props", tags=["props"])


@router.get("/players", response_model=list[str])
def list_players() -> list[str]:
    try:
        return model_service.get_player_names()
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to load players: {exc}"
        ) from exc


@router.post("/predict", response_model=PlayerPropPredictResponse)
def predict_player_prop(payload: PlayerPropPredictRequest) -> PlayerPropPredictResponse:
    try:
        result = model_service.predict_player_prop(
            player=payload.player,
            prop_type=payload.line_type,
            side=payload.side,
            line=payload.line,
            odds=payload.odds,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to score player prop: {exc}"
        ) from exc
    return PlayerPropPredictResponse(**result)
