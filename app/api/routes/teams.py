from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.services import team_service

router = APIRouter(prefix="/teams", tags=["teams"])
system_router = APIRouter()
logger = logging.getLogger(__name__)


@system_router.get("/health", tags=["health"])
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@system_router.get("/", tags=["health"])
def root() -> dict[str, str]:
    return {"message": "NBA Prediction API running"}


@router.get("/", response_model=list[str])
def list_teams() -> list[str]:
    try:
        return team_service.get_modern_team_names()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to load teams")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load teams: {exc}",
        ) from exc
