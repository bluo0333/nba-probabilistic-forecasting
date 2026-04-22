from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from app.schemas.request import MatchupPredictRequest
from app.schemas.response import MatchupPredictResponse
from app.services import model_service

router = APIRouter(prefix="/predict", tags=["predict"])
logger = logging.getLogger(__name__)


def _run_matchup_prediction(
    home: str,
    away: str,
    game_date: str | None = None,
) -> MatchupPredictResponse:
    try:
        logger.info(
            "matchup_prediction_request home=%s away=%s game_date=%s",
            home,
            away,
            game_date,
        )
        result = model_service.predict_matchup(home, away, game_date)
        return MatchupPredictResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected matchup prediction failure")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected prediction error: {exc}",
        ) from exc


@router.post("/matchup", response_model=MatchupPredictResponse)
def predict_matchup(payload: MatchupPredictRequest) -> MatchupPredictResponse:
    return _run_matchup_prediction(payload.home, payload.away, payload.game_date)


@router.get("/quick", response_model=MatchupPredictResponse)
def predict_matchup_quick(
    home: str = Query(..., min_length=1),
    away: str = Query(..., min_length=1),
    game_date: str | None = Query(default=None),
) -> MatchupPredictResponse:
    return _run_matchup_prediction(home, away, game_date)
