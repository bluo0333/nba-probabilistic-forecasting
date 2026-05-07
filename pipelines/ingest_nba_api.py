from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from nba_api.stats.endpoints import playergamelogs
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "nba_api is required. Install it with: pip install nba_api"
    ) from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
TEAMS_PATH = REPO_ROOT / "data" / "teams.csv"
OUTPUT_PATH = REPO_ROOT / "data" / "player_game_stats.csv"

REQUIRED_OUTPUT_COLUMNS = [
    "player_id",
    "player_name",
    "game_id",
    "date",
    "minutes",
    "points",
    "assists",
    "rebounds",
    "threes_made",
    "team_id",
    "opponent_team_id",
]


def default_seasons() -> list[str]:
    """Return the three most recent NBA season labels."""
    now = datetime.now()
    active_start_year = now.year if now.month >= 10 else now.year - 1
    return [
        format_season(start_year)
        for start_year in range(active_start_year - 2, active_start_year + 1)
    ]


def format_season(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def parse_season(value: str) -> str:
    """Accept either 2024 or 2024-25 and return an NBA API season label."""
    text = str(value).strip()
    if "-" in text:
        return text
    try:
        return format_season(int(text))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid season '{value}'. Use 2024 or 2024-25."
        ) from exc


def load_team_id_by_abbreviation(path: Path = TEAMS_PATH) -> dict[str, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Teams file not found: {path}")

    teams = pd.read_csv(path)
    required = {"team_id", "team_abbreviation"}
    missing = required.difference(teams.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    mapping: dict[str, int] = {}
    for row in teams.to_dict("records"):
        abbr = str(row.get("team_abbreviation") or "").strip().upper()
        if not abbr:
            continue
        try:
            team_id = int(row["team_id"])
        except (TypeError, ValueError):
            continue
        mapping.setdefault(abbr, team_id)

    if not mapping:
        raise ValueError(f"No team abbreviation mapping found in {path}.")
    return mapping


def parse_minutes(value: Any) -> float:
    """Parse minutes from nba_api values such as '34:21', '34', or None.

    nba_api returns MIN as a colon-delimited string (e.g. '34:21'), NOT a plain
    float. safe_number() would fail silently and return 0.0 for every row.
    This function handles both formats correctly.
    """
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    if ":" in text:
        parts = text.split(":", 1)
        try:
            mins = float(parts[0])
            secs = float(parts[1])
            return mins + secs / 60.0
        except (ValueError, IndexError):
            return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def safe_int(value: Any) -> int:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def parse_opponent_abbreviation(matchup: Any) -> str:
    """Extract opponent abbreviation from strings like 'BOS vs. NYK' or 'BOS @ NYK'."""
    text = str(matchup or "").strip().upper()
    for delimiter in (" VS. ", " @ "):
        if delimiter in text:
            return text.split(delimiter, 1)[1].strip()
    return ""


def normalize_rows(
    frame: pd.DataFrame,
    team_id_by_abbreviation: dict[str, int],
) -> pd.DataFrame:
    required = {
        "PLAYER_ID",
        "PLAYER_NAME",
        "GAME_ID",
        "GAME_DATE",
        "MIN",
        "PTS",
        "AST",
        "REB",
        "FG3M",
        "TEAM_ID",
        "MATCHUP",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"nba_api response missing columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        opponent_abbr = parse_opponent_abbreviation(row.get("MATCHUP"))
        rows.append(
            {
                "player_id": safe_int(row.get("PLAYER_ID")),
                "player_name": str(row.get("PLAYER_NAME") or "").strip(),
                "game_id": str(row.get("GAME_ID") or "").strip(),
                "date": pd.to_datetime(row.get("GAME_DATE"), errors="coerce"),
                # FIX: nba_api returns MIN as "34:21" string, not a plain float.
                # parse_minutes() handles both colon-format and plain floats.
                "minutes": parse_minutes(row.get("MIN")),
                "points": safe_int(row.get("PTS")),
                "assists": safe_int(row.get("AST")),
                "rebounds": safe_int(row.get("REB")),
                "threes_made": safe_int(row.get("FG3M")),
                "team_id": safe_int(row.get("TEAM_ID")),
                "opponent_team_id": team_id_by_abbreviation.get(opponent_abbr, 0),
            }
        )

    out = pd.DataFrame(rows, columns=REQUIRED_OUTPUT_COLUMNS)
    out = out[
        (out["player_id"] != 0)
        & out["player_name"].ne("")
        & out["game_id"].ne("")
    ]
    out = out.dropna(subset=["date"])
    return out


def fetch_season(
    season: str,
    team_id_by_abbreviation: dict[str, int],
    timeout: int,
) -> pd.DataFrame:
    print(f"Fetching NBA.com player game logs for season {season}...")
    endpoint = playergamelogs.PlayerGameLogs(
        season_nullable=season,
        season_type_nullable="Regular Season",
        timeout=timeout,
    )
    frames = endpoint.get_data_frames()
    if not frames or frames[0].empty:
        print(f"  No data returned for season {season}.")
        return pd.DataFrame(columns=REQUIRED_OUTPUT_COLUMNS)
    df = normalize_rows(frames[0], team_id_by_abbreviation)
    return df


def fetch_all_seasons(
    seasons: list[str],
    sleep_seconds: float,
    timeout: int,
) -> pd.DataFrame:
    team_id_by_abbreviation = load_team_id_by_abbreviation()
    frames: list[pd.DataFrame] = []

    for index, season in enumerate(seasons):
        frame = fetch_season(season, team_id_by_abbreviation, timeout=timeout)
        print(f"  → {len(frame)} rows")
        frames.append(frame)
        if index < len(seasons) - 1 and sleep_seconds > 0:
            print(f"  Sleeping {sleep_seconds}s to avoid rate limiting...")
            time.sleep(sleep_seconds)

    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        raise RuntimeError(
            "No player game logs were fetched from NBA.com. "
            "Check your internet connection or try again — NBA.com can rate-limit."
        )

    combined = combined.sort_values(["date", "game_id", "player_id"])
    combined = combined.drop_duplicates(subset=["player_id", "game_id"], keep="last")
    return combined[REQUIRED_OUTPUT_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest NBA player game logs from the free nba_api package (NBA.com).",
    )
    parser.add_argument(
        "--season",
        action="append",
        type=parse_season,
        dest="seasons",
        help=(
            "Season to fetch, e.g. 2023 or 2023-24. "
            "Repeat for multiple seasons. Defaults to last 3 seasons."
        ),
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.5,
        help="Seconds to sleep between season requests (default: 1.5).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=90,
        help="Request timeout in seconds (default: 90).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seasons = args.seasons or default_seasons()
    print(f"Using seasons: {seasons}")

    df = fetch_all_seasons(
        seasons=seasons,
        sleep_seconds=args.sleep,
        timeout=args.timeout,
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows → {OUTPUT_PATH}")
    print("Next steps:")
    print("  python pipelines/build_player_features.py")
    print("  python pipelines/train_player_model.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: NBA API ingestion failed: {exc}")
        raise SystemExit(1)