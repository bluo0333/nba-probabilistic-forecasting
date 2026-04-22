from __future__ import annotations

from typing import Any

import duckdb

from pipelines import predict_matchup as matchup_pipeline

EASTERN_CONFERENCE = {
    "ATL",
    "BKN",
    "BOS",
    "CHA",
    "CHI",
    "CLE",
    "DET",
    "IND",
    "MIA",
    "MIL",
    "NYK",
    "ORL",
    "PHI",
    "TOR",
    "WAS",
}


def team_logo_url(team_id: str) -> str:
    return f"https://cdn.nba.com/logos/nba/{team_id}/global/L/logo.svg"


def team_conference(team_abbreviation: str) -> str:
    return "East" if team_abbreviation in EASTERN_CONFERENCE else "West"


def get_teams(conn: duckdb.DuckDBPyConnection) -> list[dict[str, str]]:
    rows = conn.execute("""
        SELECT
            CAST(id AS VARCHAR) AS id,
            CAST(abbreviation AS VARCHAR) AS abbreviation,
            CAST(full_name AS VARCHAR) AS full_name
        FROM team
        ORDER BY abbreviation
        """).fetchall()
    return [
        {
            "id": team_id,
            "abbreviation": abbreviation,
            "full_name": full_name,
            "conference": team_conference(abbreviation),
            "logo_url": team_logo_url(team_id),
        }
        for team_id, abbreviation, full_name in rows
    ]


def resolve_team_id(conn: duckdb.DuckDBPyConnection, team_query: str) -> str:
    return str(matchup_pipeline.resolve_team_id(conn, team_query))


def get_team_metadata(
    conn: duckdb.DuckDBPyConnection,
    home_id: str,
    away_id: str,
) -> tuple[dict[str, str], dict[str, str]]:
    teams = conn.execute(
        """
        SELECT
            CAST(id AS VARCHAR) AS id,
            CAST(full_name AS VARCHAR) AS full_name,
            CAST(abbreviation AS VARCHAR) AS abbreviation
        FROM team
        WHERE CAST(id AS VARCHAR) IN (?, ?)
        """,
        [home_id, away_id],
    ).df()

    if teams.empty or len(teams) < 2:
        raise ValueError("Missing team metadata for one or both teams.")

    home_rows = teams[teams["id"] == home_id]
    away_rows = teams[teams["id"] == away_id]
    if home_rows.empty or away_rows.empty:
        raise ValueError("Could not resolve full team metadata for prediction.")

    home_row = home_rows.iloc[0]
    away_row = away_rows.iloc[0]
    return (
        {
            "id": str(home_row["id"]),
            "full_name": str(home_row["full_name"]),
            "abbreviation": str(home_row["abbreviation"]),
        },
        {
            "id": str(away_row["id"]),
            "full_name": str(away_row["full_name"]),
            "abbreviation": str(away_row["abbreviation"]),
        },
    )


def get_latest_team_state(
    conn: duckdb.DuckDBPyConnection, team_id: str
) -> dict[str, Any]:
    return matchup_pipeline.latest_team_state(conn, team_id)


def get_last_game_date(conn: duckdb.DuckDBPyConnection, team_id: str):
    return matchup_pipeline.last_game_date(conn, team_id)
