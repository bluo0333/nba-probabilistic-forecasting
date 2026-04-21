from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.db.connection import get_connection
from app.services import data_service

router = APIRouter(prefix="/teams", tags=["teams"])


@router.get("/", response_model=list[str])
def list_teams() -> list[str]:
    try:
        with get_connection() as conn:
            teams = data_service.get_teams(conn)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load teams: {exc}") from exc

    if not teams:
        raise HTTPException(status_code=404, detail="No teams found in local database.")

    return [team["full_name"] for team in teams]
