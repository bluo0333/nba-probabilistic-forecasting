from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd

from app.core.config import TEAM_STATES_PATH, TEAMS_PATH

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

CURRENT_NBA_TEAM_NAMES = [
    "Atlanta Hawks",
    "Boston Celtics",
    "Brooklyn Nets",
    "Charlotte Hornets",
    "Chicago Bulls",
    "Cleveland Cavaliers",
    "Dallas Mavericks",
    "Denver Nuggets",
    "Detroit Pistons",
    "Golden State Warriors",
    "Houston Rockets",
    "Indiana Pacers",
    "Los Angeles Clippers",
    "Los Angeles Lakers",
    "Memphis Grizzlies",
    "Miami Heat",
    "Milwaukee Bucks",
    "Minnesota Timberwolves",
    "New Orleans Pelicans",
    "New York Knicks",
    "Oklahoma City Thunder",
    "Orlando Magic",
    "Philadelphia 76ers",
    "Phoenix Suns",
    "Portland Trail Blazers",
    "Sacramento Kings",
    "San Antonio Spurs",
    "Toronto Raptors",
    "Utah Jazz",
    "Washington Wizards",
]

STATE_COLUMNS = [
    "elo_pre",
    "avg_pts_for_last5",
    "avg_pts_against_last5",
    "win_pct_last5",
    "netrtg_last10",
    "efg_last10",
    "tov_pct_last10",
    "orb_pct_last10",
    "ftr_last10",
]

TEAM_STATE_REQUIRED_COLUMNS = ["team_id", "last_game_date"] + STATE_COLUMNS
_PREDICTION_STATE_BY_TEAM: dict[str, dict[str, Any]] | None = None


def _normalize_text(value: str) -> str:
    return str(value or "").strip().lower()


def _normalize_team_id(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _to_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


@lru_cache(maxsize=1)
def _load_team_states() -> dict[str, dict[str, Any]]:
    if not TEAM_STATES_PATH.is_file():
        raise FileNotFoundError(f"Data file not found: {TEAM_STATES_PATH}")

    states_df = pd.read_csv(TEAM_STATES_PATH)
    missing = [column for column in TEAM_STATE_REQUIRED_COLUMNS if column not in states_df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {TEAM_STATES_PATH}: {', '.join(missing)}"
        )

    states_df["team_id"] = states_df["team_id"].map(_normalize_team_id)
    states_df["last_game_date"] = pd.to_datetime(states_df["last_game_date"], errors="coerce")
    states_df = states_df[(states_df["team_id"] != "") & states_df["last_game_date"].notna()]
    if states_df.empty:
        raise ValueError(f"No valid team states found in {TEAM_STATES_PATH}.")

    states_by_team: dict[str, dict[str, Any]] = {}
    for row in states_df.to_dict("records"):
        team_id = str(row["team_id"])
        states_by_team[team_id] = {
            "last_game_date": pd.Timestamp(row["last_game_date"]),
        }
        for column in STATE_COLUMNS:
            states_by_team[team_id][column] = _to_float(row.get(column))
    return states_by_team


def preload_prediction_data() -> None:
    global _PREDICTION_STATE_BY_TEAM
    if _PREDICTION_STATE_BY_TEAM is not None:
        return

    _ensure_team_index()
    _PREDICTION_STATE_BY_TEAM = _load_team_states()


@lru_cache(maxsize=1)
def _team_index_bundle() -> tuple[dict[str, dict[str, str]], dict[str, tuple[str, ...]]]:
    if not TEAMS_PATH.is_file():
        raise FileNotFoundError(f"Data file not found: {TEAMS_PATH}")

    teams_df = pd.read_csv(TEAMS_PATH)
    if "team_id" not in teams_df.columns or "team_name" not in teams_df.columns:
        raise ValueError(
            f"{TEAMS_PATH} must include columns: team_id, team_name"
        )

    abbreviation_column = "team_abbreviation" if "team_abbreviation" in teams_df.columns else None
    teams_by_id: dict[str, dict[str, str]] = {}
    alias_to_ids_mut: dict[str, set[str]] = {}

    for row in teams_df.to_dict("records"):
        team_id = _normalize_team_id(row.get("team_id"))
        team_name = str(row.get("team_name") or "").strip()
        if not team_id or not team_name:
            continue

        abbreviation = ""
        if abbreviation_column is not None:
            abbreviation = str(row.get(abbreviation_column) or "").strip().upper()

        teams_by_id[team_id] = {
            "id": team_id,
            "abbreviation": abbreviation,
            "full_name": team_name,
        }

        aliases = {
            _normalize_text(team_id),
            _normalize_text(team_name),
        }
        if abbreviation:
            aliases.add(_normalize_text(abbreviation))

        for alias in aliases:
            if not alias:
                continue
            alias_to_ids_mut.setdefault(alias, set()).add(team_id)

    if not teams_by_id:
        raise ValueError(f"No valid teams found in {TEAMS_PATH}.")

    alias_to_ids = {
        key: tuple(sorted(value))
        for key, value in alias_to_ids_mut.items()
    }
    return teams_by_id, alias_to_ids


@lru_cache(maxsize=1)
def _teams_payload() -> tuple[dict[str, str], ...]:
    teams = list(_ensure_team_index().values())
    teams.sort(key=lambda row: (row["abbreviation"] or row["full_name"], row["full_name"]))
    return tuple(
        {
            "id": team["id"],
            "abbreviation": team["abbreviation"],
            "full_name": team["full_name"],
            "conference": team_conference(team["abbreviation"]),
            "logo_url": team_logo_url(team["id"]),
        }
        for team in teams
    )


@lru_cache(maxsize=1)
def _modern_team_names() -> tuple[str, ...]:
    all_names = {
        str(team["full_name"]).strip()
        for team in _teams_payload()
        if team.get("full_name")
    }
    modern_names = sorted(name for name in all_names if name in CURRENT_NBA_TEAM_NAMES)
    return tuple(modern_names if modern_names else sorted(all_names))


def preload_team_index() -> None:
    _team_index_bundle()
    _teams_payload()
    _modern_team_names()


def _ensure_team_index() -> dict[str, dict[str, str]]:
    teams_by_id, _ = _team_index_bundle()
    return dict(teams_by_id)


def team_logo_url(team_id: str) -> str:
    return f"https://cdn.nba.com/logos/nba/{team_id}/global/L/logo.svg"


def team_conference(team_abbreviation: str) -> str:
    return "East" if team_abbreviation in EASTERN_CONFERENCE else "West"


def get_teams() -> list[dict[str, str]]:
    return [dict(team) for team in _teams_payload()]


def get_modern_nba_team_names() -> list[str]:
    return list(_modern_team_names())


def resolve_team_id(team_query: str) -> str:
    teams, alias_to_ids = _team_index_bundle()
    q = _normalize_text(team_query)
    if not q:
        raise ValueError("Team cannot be empty.")

    if q in alias_to_ids:
        exact = list(alias_to_ids[q])
        return exact[0]

    partial_ids: set[str] = set()
    for alias, ids in alias_to_ids.items():
        if q in alias:
            partial_ids.update(ids)
    partial = sorted(partial_ids)
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        names = ", ".join(sorted({teams[team_id]["full_name"] for team_id in partial}))
        raise ValueError(f"Ambiguous team '{team_query}'. Matches: {names}")

    raise ValueError(f"Team '{team_query}' not found.")


def get_team_metadata(
    home_id: str,
    away_id: str,
) -> tuple[dict[str, str], dict[str, str]]:
    teams = _ensure_team_index()
    home_meta = teams.get(str(home_id))
    away_meta = teams.get(str(away_id))
    if home_meta is None or away_meta is None:
        raise ValueError("Missing team metadata for one or both teams.")
    return (
        {
            "id": home_meta["id"],
            "full_name": home_meta["full_name"],
            "abbreviation": home_meta["abbreviation"],
        },
        {
            "id": away_meta["id"],
            "full_name": away_meta["full_name"],
            "abbreviation": away_meta["abbreviation"],
        },
    )


def get_matchup_snapshot(
    home_id: str,
    away_id: str,
    as_of_date: pd.Timestamp | None = None,
) -> tuple[dict[str, Any], dict[str, Any], pd.Timestamp, pd.Timestamp]:
    home_id = _normalize_team_id(home_id)
    away_id = _normalize_team_id(away_id)
    if not home_id or not away_id:
        raise ValueError("Both teams must be valid.")

    preload_prediction_data()
    states = _PREDICTION_STATE_BY_TEAM or {}
    try:
        home_state_src = states[home_id]
        away_state_src = states[away_id]
    except KeyError as exc:
        raise ValueError("Missing historical features for one or both teams.") from exc

    if as_of_date is not None:
        as_of = pd.Timestamp(as_of_date)
        home_last = pd.Timestamp(home_state_src["last_game_date"])
        away_last = pd.Timestamp(away_state_src["last_game_date"])
        if as_of <= home_last or as_of <= away_last:
            raise ValueError("Prediction context unavailable for requested game_date.")

    home_state = dict(home_state_src)
    away_state = dict(away_state_src)
    home_last_game = pd.Timestamp(home_state["last_game_date"])
    away_last_game = pd.Timestamp(away_state["last_game_date"])
    return home_state, away_state, home_last_game, away_last_game


def get_latest_team_state(team_id: str) -> dict[str, Any]:
    team_id_norm = _normalize_team_id(team_id)
    if not team_id_norm:
        raise ValueError("Team id is required.")
    preload_prediction_data()
    states = _PREDICTION_STATE_BY_TEAM or {}
    state = states.get(team_id_norm)
    if state is None:
        raise ValueError(f"No historical features found for team id '{team_id}'.")
    return dict(state)


def get_last_game_date(team_id: str):
    state = get_latest_team_state(team_id)
    return pd.Timestamp(state["last_game_date"])
