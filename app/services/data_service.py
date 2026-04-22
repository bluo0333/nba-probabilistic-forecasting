from __future__ import annotations

from collections import deque
import math
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_GAMES_PATH = PROJECT_ROOT / "data" / "processed_games.csv"

INITIAL_ELO = 1500.0
K = 20.0
HOME_ADVANTAGE = 100.0
CARRYOVER = 0.75

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

TEAM_COLUMNS = [
    "team_id_home",
    "team_abbreviation_home",
    "team_name_home",
    "team_id_away",
    "team_abbreviation_away",
    "team_name_away",
]
GAME_COLUMNS = [
    "season_id",
    "game_id",
    "game_date",
    "team_id_home",
    "team_id_away",
    "wl_home",
    "pts_home",
    "pts_away",
    "fga_home",
    "fta_home",
    "oreb_home",
    "tov_home",
    "fga_away",
    "fta_away",
    "oreb_away",
    "tov_away",
    "fgm_home",
    "fg3m_home",
    "dreb_away",
    "fgm_away",
    "fg3m_away",
    "dreb_home",
]
NUMERIC_COLUMNS = [
    "season_id",
    "pts_home",
    "pts_away",
    "fga_home",
    "fta_home",
    "oreb_home",
    "tov_home",
    "fga_away",
    "fta_away",
    "oreb_away",
    "tov_away",
    "fgm_home",
    "fg3m_home",
    "dreb_away",
    "fgm_away",
    "fg3m_away",
    "dreb_home",
]

_TEAM_INDEX: dict[str, dict[str, str]] | None = None
MODERN_NBA_MIN_SEASON_ID = 22018
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


def _safe_mean(values: deque[float], min_count: int) -> float:
    valid = [value for value in values if pd.notna(value)]
    if len(valid) < min_count:
        return float("nan")
    return float(sum(valid) / len(valid))


def _safe_sum(values: deque[float]) -> float:
    valid = [value for value in values if pd.notna(value)]
    if not valid:
        return float("nan")
    return float(sum(valid))


def _compute_possessions(
    fga_for: float,
    fta_for: float,
    oreb_for: float,
    tov_for: float,
    fga_against: float,
    fta_against: float,
    oreb_against: float,
    tov_against: float,
) -> float:
    values = [
        fga_for,
        fta_for,
        oreb_for,
        tov_for,
        fga_against,
        fta_against,
        oreb_against,
        tov_against,
    ]
    if any(pd.isna(value) for value in values):
        return float("nan")
    return 0.5 * (
        (fga_for + 0.44 * fta_for - oreb_for + tov_for)
        + (fga_against + 0.44 * fta_against - oreb_against + tov_against)
    )


def _parse_home_win(value: Any) -> float:
    text = str(value or "").strip().upper()
    if text == "W":
        return 1.0
    if text == "L":
        return 0.0
    return float("nan")


def _iter_processed_games(chunksize: int = 20000):
    if not PROCESSED_GAMES_PATH.is_file():
        raise FileNotFoundError(f"Data file not found: {PROCESSED_GAMES_PATH}")

    for chunk in pd.read_csv(PROCESSED_GAMES_PATH, usecols=GAME_COLUMNS, chunksize=chunksize):
        chunk["game_date"] = pd.to_datetime(chunk["game_date"], errors="coerce")
        for column in NUMERIC_COLUMNS:
            if column in chunk.columns:
                chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        chunk["team_id_home"] = chunk["team_id_home"].map(_normalize_team_id)
        chunk["team_id_away"] = chunk["team_id_away"].map(_normalize_team_id)
        chunk = chunk.dropna(subset=["game_date"])
        chunk = chunk[(chunk["team_id_home"] != "") & (chunk["team_id_away"] != "")]
        chunk = chunk.sort_values(["game_date", "game_id"])
        yield chunk


def _ensure_team_index() -> dict[str, dict[str, str]]:
    global _TEAM_INDEX
    if _TEAM_INDEX is not None:
        return _TEAM_INDEX

    if not PROCESSED_GAMES_PATH.is_file():
        raise FileNotFoundError(f"Data file not found: {PROCESSED_GAMES_PATH}")

    by_id: dict[str, dict[str, str]] = {}
    for chunk in pd.read_csv(PROCESSED_GAMES_PATH, usecols=TEAM_COLUMNS, chunksize=20000):
        for _, row in chunk.iterrows():
            home_id = _normalize_team_id(row.get("team_id_home"))
            away_id = _normalize_team_id(row.get("team_id_away"))
            if home_id and home_id not in by_id:
                by_id[home_id] = {
                    "id": home_id,
                    "abbreviation": str(row.get("team_abbreviation_home") or "").strip().upper(),
                    "full_name": str(row.get("team_name_home") or "").strip(),
                }
            if away_id and away_id not in by_id:
                by_id[away_id] = {
                    "id": away_id,
                    "abbreviation": str(row.get("team_abbreviation_away") or "").strip().upper(),
                    "full_name": str(row.get("team_name_away") or "").strip(),
                }

    _TEAM_INDEX = {k: v for k, v in by_id.items() if v["id"] and v["full_name"]}
    return _TEAM_INDEX


def team_logo_url(team_id: str) -> str:
    return f"https://cdn.nba.com/logos/nba/{team_id}/global/L/logo.svg"


def team_conference(team_abbreviation: str) -> str:
    return "East" if team_abbreviation in EASTERN_CONFERENCE else "West"


def get_teams(_conn: Any = None) -> list[dict[str, str]]:
    teams = list(_ensure_team_index().values())
    teams.sort(key=lambda row: row["abbreviation"])
    return [
        {
            "id": team["id"],
            "abbreviation": team["abbreviation"],
            "full_name": team["full_name"],
            "conference": team_conference(team["abbreviation"]),
            "logo_url": team_logo_url(team["id"]),
        }
        for team in teams
    ]


def get_modern_nba_team_names(conn: Any) -> list[str]:
    has_game_table = conn.execute(
        """
        SELECT 1
        FROM information_schema.tables
        WHERE lower(table_name) = 'game'
        LIMIT 1
        """
    ).fetchone()

    if has_game_table:
        table_ref = "game"
    else:
        if not PROCESSED_GAMES_PATH.is_file():
            raise FileNotFoundError(f"Data file not found: {PROCESSED_GAMES_PATH}")
        csv_path = PROCESSED_GAMES_PATH.resolve().as_posix().replace("'", "''")
        table_ref = f"read_csv_auto('{csv_path}', header=true)"

    current_name_list_sql = ", ".join(
        "'" + name.replace("'", "''") + "'" for name in CURRENT_NBA_TEAM_NAMES
    )
    rows = conn.execute(
        f"""
        WITH raw_teams AS (
            SELECT trim(CAST(team_name_home AS VARCHAR)) AS team_name
            FROM {table_ref}
            WHERE CAST(season_id AS BIGINT) >= ?
            UNION ALL
            SELECT trim(CAST(team_name_away AS VARCHAR)) AS team_name
            FROM {table_ref}
            WHERE CAST(season_id AS BIGINT) >= ?
        ),
        teams AS (
            SELECT
                CASE
                    WHEN team_name = 'LA Clippers' THEN 'Los Angeles Clippers'
                    WHEN team_name = 'New Jersey Nets' THEN 'Brooklyn Nets'
                    WHEN team_name = 'New Orleans Hornets' THEN 'New Orleans Pelicans'
                    WHEN team_name = 'Seattle SuperSonics' THEN 'Oklahoma City Thunder'
                    ELSE team_name
                END AS team_name
            FROM raw_teams
        )
        SELECT DISTINCT team_name
        FROM teams
        WHERE team_name IS NOT NULL
          AND team_name <> ''
          AND team_name IN ({current_name_list_sql})
        ORDER BY team_name ASC
        """,
        [MODERN_NBA_MIN_SEASON_ID, MODERN_NBA_MIN_SEASON_ID],
    ).fetchall()
    return [str(row[0]) for row in rows]


def resolve_team_id(_conn: Any, team_query: str) -> str:
    teams = _ensure_team_index()
    q = _normalize_text(team_query)
    if not q:
        raise ValueError("Team cannot be empty.")

    exact = [
        team["id"]
        for team in teams.values()
        if q in {
            _normalize_text(team["id"]),
            _normalize_text(team["abbreviation"]),
            _normalize_text(team["full_name"]),
        }
    ]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        names = ", ".join(sorted({teams[team_id]["full_name"] for team_id in exact}))
        raise ValueError(f"Ambiguous team '{team_query}'. Matches: {names}")

    partial = [
        team["id"]
        for team in teams.values()
        if q in _normalize_text(team["full_name"]) or q in _normalize_text(team["abbreviation"])
    ]
    partial = sorted(set(partial))
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        names = ", ".join(sorted({teams[team_id]["full_name"] for team_id in partial}))
        raise ValueError(f"Ambiguous team '{team_query}'. Matches: {names}")

    raise ValueError(f"Team '{team_query}' not found.")


def get_team_metadata(
    _conn: Any,
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


def _init_tracker() -> dict[str, Any]:
    return {
        "elo": INITIAL_ELO,
        "last_game_date": None,
        "pts_for_last5": deque(maxlen=5),
        "pts_against_last5": deque(maxlen=5),
        "win_last5": deque(maxlen=5),
        "pf_last10": deque(maxlen=10),
        "pa_last10": deque(maxlen=10),
        "poss_last10": deque(maxlen=10),
        "fgm_last10": deque(maxlen=10),
        "fg3m_last10": deque(maxlen=10),
        "fga_last10": deque(maxlen=10),
        "fta_last10": deque(maxlen=10),
        "oreb_last10": deque(maxlen=10),
        "tov_last10": deque(maxlen=10),
        "opp_dreb_last10": deque(maxlen=10),
    }


def _tracker_to_state(tracker: dict[str, Any]) -> dict[str, Any]:
    pf10 = _safe_sum(tracker["pf_last10"])
    pa10 = _safe_sum(tracker["pa_last10"])
    poss10 = _safe_sum(tracker["poss_last10"])

    netrtg_last10 = float("nan")
    if pd.notna(pf10) and pd.notna(pa10) and pd.notna(poss10) and poss10 > 0:
        if len([v for v in tracker["poss_last10"] if pd.notna(v)]) >= 5:
            ortg = 100.0 * pf10 / poss10
            drtg = 100.0 * pa10 / poss10
            netrtg_last10 = ortg - drtg

    fgm10 = _safe_sum(tracker["fgm_last10"])
    fg3m10 = _safe_sum(tracker["fg3m_last10"])
    fga10 = _safe_sum(tracker["fga_last10"])
    fta10 = _safe_sum(tracker["fta_last10"])
    oreb10 = _safe_sum(tracker["oreb_last10"])
    tov10 = _safe_sum(tracker["tov_last10"])
    opp_dreb10 = _safe_sum(tracker["opp_dreb_last10"])

    efg_last10 = float("nan")
    if pd.notna(fgm10) and pd.notna(fg3m10) and pd.notna(fga10) and fga10 > 0:
        if len([v for v in tracker["fga_last10"] if pd.notna(v)]) >= 5:
            efg_last10 = (fgm10 + 0.5 * fg3m10) / fga10

    tov_pct_last10 = float("nan")
    tov_denom = (
        (fga10 if pd.notna(fga10) else float("nan"))
        + 0.44 * (fta10 if pd.notna(fta10) else float("nan"))
        + (tov10 if pd.notna(tov10) else float("nan"))
    )
    if pd.notna(tov10) and pd.notna(tov_denom) and tov_denom > 0:
        if len([v for v in tracker["tov_last10"] if pd.notna(v)]) >= 5:
            tov_pct_last10 = tov10 / tov_denom

    orb_pct_last10 = float("nan")
    orb_denom = (
        (oreb10 if pd.notna(oreb10) else float("nan"))
        + (opp_dreb10 if pd.notna(opp_dreb10) else float("nan"))
    )
    if pd.notna(oreb10) and pd.notna(opp_dreb10) and pd.notna(orb_denom) and orb_denom > 0:
        if len([v for v in tracker["oreb_last10"] if pd.notna(v)]) >= 5:
            orb_pct_last10 = oreb10 / orb_denom

    ftr_last10 = float("nan")
    if pd.notna(fta10) and pd.notna(fga10) and fga10 > 0:
        if len([v for v in tracker["fta_last10"] if pd.notna(v)]) >= 5:
            ftr_last10 = fta10 / fga10

    return {
        "elo_pre": float(tracker["elo"]),
        "avg_pts_for_last5": _safe_mean(tracker["pts_for_last5"], min_count=3),
        "avg_pts_against_last5": _safe_mean(tracker["pts_against_last5"], min_count=3),
        "win_pct_last5": _safe_mean(tracker["win_last5"], min_count=3),
        "netrtg_last10": netrtg_last10,
        "efg_last10": efg_last10,
        "tov_pct_last10": tov_pct_last10,
        "orb_pct_last10": orb_pct_last10,
        "ftr_last10": ftr_last10,
        "last_game_date": tracker["last_game_date"],
    }


def get_matchup_snapshot(
    home_id: str,
    away_id: str,
    as_of_date: pd.Timestamp | None = None,
) -> tuple[dict[str, Any], dict[str, Any], pd.Timestamp, pd.Timestamp]:
    home_id = _normalize_team_id(home_id)
    away_id = _normalize_team_id(away_id)
    if not home_id or not away_id:
        raise ValueError("Both teams must be valid.")

    ratings: dict[str, float] = {}
    trackers: dict[str, dict[str, Any]] = {}
    current_season: float | None = None

    for chunk in _iter_processed_games():
        for _, row in chunk.iterrows():
            game_date = row["game_date"]
            if as_of_date is not None and game_date >= as_of_date:
                continue

            season = row.get("season_id")
            if (
                current_season is not None
                and pd.notna(season)
                and season != current_season
            ):
                for team in ratings:
                    ratings[team] = CARRYOVER * ratings[team] + (1.0 - CARRYOVER) * INITIAL_ELO
            if pd.notna(season):
                current_season = float(season)

            row_home_id = _normalize_team_id(row.get("team_id_home"))
            row_away_id = _normalize_team_id(row.get("team_id_away"))
            if not row_home_id or not row_away_id:
                continue

            if row_home_id not in trackers:
                trackers[row_home_id] = _init_tracker()
            if row_away_id not in trackers:
                trackers[row_away_id] = _init_tracker()
            ratings.setdefault(row_home_id, INITIAL_ELO)
            ratings.setdefault(row_away_id, INITIAL_ELO)

            home_elo = ratings[row_home_id]
            away_elo = ratings[row_away_id]

            home_win = _parse_home_win(row.get("wl_home"))
            if pd.isna(home_win):
                continue

            pts_home = _to_float(row.get("pts_home"))
            pts_away = _to_float(row.get("pts_away"))
            point_diff = abs((pts_home if pd.notna(pts_home) else 0.0) - (pts_away if pd.notna(pts_away) else 0.0))
            expected_home = 1.0 / (1.0 + 10.0 ** ((away_elo - (home_elo + HOME_ADVANTAGE)) / 400.0))
            mov_multiplier = math.log(point_diff + 1.0) * (2.2 / ((home_elo - away_elo) * 0.001 + 2.2))

            ratings[row_home_id] = home_elo + K * mov_multiplier * (home_win - expected_home)
            ratings[row_away_id] = away_elo + K * mov_multiplier * ((1.0 - home_win) - (1.0 - expected_home))

            home_tracker = trackers[row_home_id]
            away_tracker = trackers[row_away_id]
            home_tracker["elo"] = ratings[row_home_id]
            away_tracker["elo"] = ratings[row_away_id]
            home_tracker["last_game_date"] = game_date
            away_tracker["last_game_date"] = game_date

            home_tracker["pts_for_last5"].append(pts_home)
            home_tracker["pts_against_last5"].append(pts_away)
            home_tracker["win_last5"].append(home_win)

            away_tracker["pts_for_last5"].append(pts_away)
            away_tracker["pts_against_last5"].append(pts_home)
            away_tracker["win_last5"].append(1.0 - home_win)

            home_poss = _compute_possessions(
                _to_float(row.get("fga_home")),
                _to_float(row.get("fta_home")),
                _to_float(row.get("oreb_home")),
                _to_float(row.get("tov_home")),
                _to_float(row.get("fga_away")),
                _to_float(row.get("fta_away")),
                _to_float(row.get("oreb_away")),
                _to_float(row.get("tov_away")),
            )
            away_poss = home_poss

            home_tracker["pf_last10"].append(pts_home)
            home_tracker["pa_last10"].append(pts_away)
            home_tracker["poss_last10"].append(home_poss)
            away_tracker["pf_last10"].append(pts_away)
            away_tracker["pa_last10"].append(pts_home)
            away_tracker["poss_last10"].append(away_poss)

            home_tracker["fgm_last10"].append(_to_float(row.get("fgm_home")))
            home_tracker["fg3m_last10"].append(_to_float(row.get("fg3m_home")))
            home_tracker["fga_last10"].append(_to_float(row.get("fga_home")))
            home_tracker["fta_last10"].append(_to_float(row.get("fta_home")))
            home_tracker["oreb_last10"].append(_to_float(row.get("oreb_home")))
            home_tracker["tov_last10"].append(_to_float(row.get("tov_home")))
            home_tracker["opp_dreb_last10"].append(_to_float(row.get("dreb_away")))

            away_tracker["fgm_last10"].append(_to_float(row.get("fgm_away")))
            away_tracker["fg3m_last10"].append(_to_float(row.get("fg3m_away")))
            away_tracker["fga_last10"].append(_to_float(row.get("fga_away")))
            away_tracker["fta_last10"].append(_to_float(row.get("fta_away")))
            away_tracker["oreb_last10"].append(_to_float(row.get("oreb_away")))
            away_tracker["tov_last10"].append(_to_float(row.get("tov_away")))
            away_tracker["opp_dreb_last10"].append(_to_float(row.get("dreb_home")))

    if home_id not in trackers or away_id not in trackers:
        raise ValueError("Missing historical features for one or both teams.")

    home_state = _tracker_to_state(trackers[home_id])
    away_state = _tracker_to_state(trackers[away_id])

    home_last_game = home_state.get("last_game_date")
    away_last_game = away_state.get("last_game_date")
    if home_last_game is None or away_last_game is None:
        raise ValueError("No game history found for one or both teams.")

    return (
        home_state,
        away_state,
        pd.Timestamp(home_last_game),
        pd.Timestamp(away_last_game),
    )


def get_latest_team_state(_conn: Any, team_id: str) -> dict[str, Any]:
    state, _, _, _ = get_matchup_snapshot(team_id, team_id)
    return state


def get_last_game_date(_conn: Any, team_id: str):
    state, _, _, _ = get_matchup_snapshot(team_id, team_id)
    return pd.Timestamp(state["last_game_date"])
