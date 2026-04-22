from __future__ import annotations

from app.services import data_service


def get_modern_team_names() -> list[str]:
    teams = data_service.get_modern_nba_team_names()
    if not teams:
        raise ValueError("No teams found in local dataset.")
    return teams
