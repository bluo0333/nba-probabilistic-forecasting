from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests


BASE_URL = "https://api.balldontlie.io/v1/stats"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "player_game_stats.csv"


def build_headers() -> dict[str, str]:
    """Build request headers using optional API key from environment."""
    api_key = os.getenv("BALLDONTLIE_API_KEY", "").strip()
    if not api_key:
        return {}
    return {"Authorization": api_key}


def parse_minutes(value: Any) -> float:
    """Parse minutes from API values such as '34', '34:21', or None."""
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    if ":" in text:
        minute_part, second_part = text.split(":", 1)
        try:
            minutes = float(minute_part)
            seconds = float(second_part)
            return minutes + (seconds / 60.0)
        except ValueError:
            return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def safe_int(value: Any) -> int:
    """Convert numeric-like API values to int with 0 fallback."""
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def infer_opponent_team_id(game: dict[str, Any], team_id: int) -> int:
    """Infer opponent team id from home/visitor ids and player's team id."""
    home_id = safe_int(game.get("home_team_id"))
    visitor_id = safe_int(game.get("visitor_team_id"))
    if team_id == home_id:
        return visitor_id
    if team_id == visitor_id:
        return home_id
    return 0


def normalize_stat_row(item: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize one balldontlie stats row to the required output schema."""
    player = item.get("player") or {}
    game = item.get("game") or {}
    team = item.get("team") or {}

    player_id = safe_int(player.get("id"))
    first_name = str(player.get("first_name") or "").strip()
    last_name = str(player.get("last_name") or "").strip()
    player_name = f"{first_name} {last_name}".strip()
    game_id = safe_int(game.get("id"))

    raw_date = game.get("date") or item.get("date")
    if player_id == 0 or game_id == 0 or not player_name or not raw_date:
        return None

    team_id = safe_int(team.get("id") or item.get("team_id"))
    opponent_team_id = infer_opponent_team_id(game, team_id)

    return {
        "player_id": player_id,
        "player_name": player_name,
        "game_id": game_id,
        "date": pd.to_datetime(raw_date, utc=False, errors="coerce"),
        "minutes": parse_minutes(item.get("min")),
        "points": safe_int(item.get("pts")),
        "assists": safe_int(item.get("ast")),
        "rebounds": safe_int(item.get("reb")),
        "team_id": team_id,
        "opponent_team_id": opponent_team_id,
    }


def extract_next_page(meta: dict[str, Any]) -> tuple[int | None, str | None]:
    """Extract pagination pointers from varying balldontlie meta formats."""
    next_page = meta.get("next_page")
    next_cursor = meta.get("next_cursor") or meta.get("next_cursor_id")

    parsed_next_page = None
    if next_page is not None:
        try:
            parsed_next_page = int(next_page)
        except (TypeError, ValueError):
            parsed_next_page = None

    parsed_next_cursor = None
    if next_cursor is not None and str(next_cursor).strip():
        parsed_next_cursor = str(next_cursor).strip()

    return parsed_next_page, parsed_next_cursor


def fetch_stats_for_season(
    session: requests.Session,
    season: int,
    per_page: int = 100,
) -> list[dict[str, Any]]:
    """Fetch all player game stat rows for one season."""
    rows: list[dict[str, Any]] = []
    page: int | None = 1
    cursor: str | None = None
    seen_tokens: set[tuple[int | None, str | None]] = set()

    while True:
        params: dict[str, Any] = {"seasons[]": season, "per_page": per_page}
        if cursor:
            params["cursor"] = cursor
        elif page is not None:
            params["page"] = page

        response = session.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") or []
        if not data:
            break

        for item in data:
            row = normalize_stat_row(item)
            if row is not None:
                rows.append(row)

        meta = payload.get("meta") or {}
        next_page, next_cursor = extract_next_page(meta)
        token = (next_page, next_cursor)
        if token in seen_tokens:
            break
        seen_tokens.add(token)

        if next_cursor:
            cursor = next_cursor
            page = None
            continue
        if next_page:
            page = next_page
            cursor = None
            continue
        break

    return rows


def fetch_all_stats(seasons: list[int]) -> pd.DataFrame:
    """Fetch and combine stats across all requested seasons."""
    headers = build_headers()
    with requests.Session() as session:
        if headers:
            session.headers.update(headers)

        all_rows: list[dict[str, Any]] = []
        for season in seasons:
            print(f"Fetching balldontlie stats for season {season}...")
            season_rows = fetch_stats_for_season(session, season=season, per_page=100)
            print(f"Season {season}: fetched {len(season_rows)} rows")
            all_rows.extend(season_rows)

    if not all_rows:
        raise RuntimeError(
            "No stats were fetched from balldontlie. "
            "Check API availability, API key, and requested seasons."
        )

    df = pd.DataFrame(all_rows)
    df = df.dropna(subset=["date"])
    df = df.sort_values(["date", "game_id", "player_id"]).drop_duplicates(
        subset=["player_id", "game_id"],
        keep="last",
    )
    return df


def default_seasons() -> list[int]:
    """Return a default range of recent seasons."""
    current_year = datetime.now().year
    return [current_year - 2, current_year - 1, current_year]


def main() -> None:
    seasons = default_seasons()
    print(f"Using seasons: {seasons}")
    df = fetch_all_stats(seasons=seasons)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except requests.HTTPError as exc:
        print(f"ERROR: balldontlie request failed: {exc}")
        raise SystemExit(1)
    except Exception as exc:
        print(f"ERROR: ingestion failed: {exc}")
        raise SystemExit(1)
