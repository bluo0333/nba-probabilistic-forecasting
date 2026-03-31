from __future__ import annotations

import argparse
import os
import re
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import requests


BASE_URL = "https://api.balldontlie.io/v1/games"
REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = REPO_ROOT / "data" / "nba.duckdb"
LIVE_GAMES_TABLE = "live_games"
TABLE_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def build_headers() -> dict[str, str]:
    api_key = os.getenv("BALLDONTLIE_API_KEY", "").strip()
    if not api_key:
        return {}
    return {"Authorization": api_key}


def extract_next_page(meta: dict[str, Any]) -> tuple[int | None, str | None]:
    next_page = meta.get("next_page")
    next_cursor = meta.get("next_cursor") or meta.get("next_cursor_id")

    parsed_next_page: int | None = None
    if next_page is not None:
        try:
            parsed_next_page = int(next_page)
        except (TypeError, ValueError):
            parsed_next_page = None

    parsed_next_cursor: str | None = None
    if next_cursor is not None and str(next_cursor).strip():
        parsed_next_cursor = str(next_cursor).strip()

    return parsed_next_page, parsed_next_cursor


def normalize_team(raw_team: dict[str, Any]) -> tuple[str, str, str]:
    team_id = str(raw_team.get("id") or "").strip()
    abbreviation = str(raw_team.get("abbreviation") or "").strip()
    full_name = str(raw_team.get("full_name") or "").strip()

    if not full_name:
        city = str(raw_team.get("city") or "").strip()
        name = str(raw_team.get("name") or "").strip()
        full_name = f"{city} {name}".strip()

    return team_id, abbreviation, full_name


def safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_game(raw_game: dict[str, Any]) -> dict[str, Any] | None:
    raw_date = raw_game.get("date")
    if raw_date is None:
        return None

    parsed_date = pd.to_datetime(raw_date, utc=False, errors="coerce")
    if pd.isna(parsed_date):
        return None

    home_team = raw_game.get("home_team") or {}
    away_team = raw_game.get("visitor_team") or raw_game.get("away_team") or {}

    home_id, home_abbrev, home_name = normalize_team(home_team)
    away_id, away_abbrev, away_name = normalize_team(away_team)
    if not home_id or not away_id:
        return None

    game_id = str(raw_game.get("id") or "").strip()
    if not game_id:
        return None

    is_postseason = bool(raw_game.get("postseason"))
    season_type = "Playoffs" if is_postseason else "Regular Season"

    return {
        "game_id": game_id,
        "game_date": parsed_date.date().isoformat(),
        "season_type": season_type,
        "team_id_away": away_id,
        "team_abbreviation_away": away_abbrev,
        "team_name_away": away_name,
        "pts_away": safe_int(raw_game.get("visitor_team_score") or raw_game.get("away_team_score")),
        "team_id_home": home_id,
        "team_abbreviation_home": home_abbrev,
        "team_name_home": home_name,
        "pts_home": safe_int(raw_game.get("home_team_score")),
    }


def fetch_games_for_date(session: requests.Session, target_date: date) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    page: int | None = 1
    cursor: str | None = None
    seen_tokens: set[tuple[int | None, str | None]] = set()

    while True:
        params: dict[str, Any] = {
            "dates[]": target_date.isoformat(),
            "per_page": 100,
        }
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
            normalized = normalize_game(item)
            if normalized is not None:
                rows.append(normalized)

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


def upsert_live_games(df: pd.DataFrame, target_date: date, db_path: Path, table_name: str) -> None:
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                game_id VARCHAR,
                game_date DATE,
                season_type VARCHAR,
                team_id_away VARCHAR,
                team_abbreviation_away VARCHAR,
                team_name_away VARCHAR,
                pts_away BIGINT,
                team_id_home VARCHAR,
                team_abbreviation_home VARCHAR,
                team_name_home VARCHAR,
                pts_home BIGINT,
                source VARCHAR,
                updated_at TIMESTAMP
            )
            """
        )

        conn.execute(
            f"DELETE FROM {table_name} WHERE CAST(game_date AS DATE) = ?",
            [target_date],
        )

        if not df.empty:
            insert_df = df.copy()
            insert_df["source"] = "balldontlie"
            insert_df["updated_at"] = pd.Timestamp.now()
            conn.register("live_games_batch", insert_df)
            conn.execute(
                f"""
                INSERT INTO {table_name}
                SELECT
                    game_id,
                    CAST(game_date AS DATE),
                    season_type,
                    team_id_away,
                    team_abbreviation_away,
                    team_name_away,
                    pts_away,
                    team_id_home,
                    team_abbreviation_home,
                    team_name_home,
                    pts_home,
                    source,
                    updated_at
                FROM live_games_batch
                """
            )
            conn.unregister("live_games_batch")
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch today's NBA schedule from balldontlie and store it in DuckDB.",
    )
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Target date in YYYY-MM-DD (default: local today).",
    )
    parser.add_argument(
        "--db-path",
        default=str(DB_PATH),
        help=f"DuckDB file path (default: {DB_PATH}).",
    )
    parser.add_argument(
        "--table",
        default=LIVE_GAMES_TABLE,
        help=f"Destination table name (default: {LIVE_GAMES_TABLE}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_date = date.fromisoformat(args.date)
    db_path = Path(args.db_path)
    table_name = str(args.table).strip() or LIVE_GAMES_TABLE
    if not TABLE_NAME_PATTERN.match(table_name):
        raise ValueError("table name must match [A-Za-z_][A-Za-z0-9_]*")

    headers = build_headers()
    with requests.Session() as session:
        if headers:
            session.headers.update(headers)
        games = fetch_games_for_date(session, target_date=target_date)

    df = pd.DataFrame(games)
    upsert_live_games(df, target_date=target_date, db_path=db_path, table_name=table_name)
    print(
        f"Saved {len(df)} game(s) for {target_date.isoformat()} "
        f"to {db_path}::{table_name}"
    )


if __name__ == "__main__":
    try:
        main()
    except requests.HTTPError as exc:
        print(f"ERROR: balldontlie request failed: {exc}")
        raise SystemExit(1)
    except ValueError as exc:
        print(f"ERROR: invalid input: {exc}")
        raise SystemExit(1)
    except Exception as exc:
        print(f"ERROR: schedule ingestion failed: {exc}")
        raise SystemExit(1)
