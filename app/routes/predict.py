from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.schemas.predict_schema import MatchupPredictRequest, MatchupPredictResponse
from app.services import model_service

router = APIRouter(prefix="/predict", tags=["predict"])


def _run_matchup_prediction(
    home: str, away: str, game_date: str | None = None
) -> MatchupPredictResponse:
    try:
        result = model_service.predict_matchup(home, away, game_date)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Unexpected prediction error: {exc}"
        ) from exc
    return MatchupPredictResponse(**result)


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
