from __future__ import annotations

from collections import deque
from pathlib import Path
import math
from typing import Any

import pandas as pd

PROCESSED_GAMES_PATH = Path("data/processed_games.csv")
FINAL_FEATURES_PATH = Path("data/final_features.csv")
TEAMS_OUTPUT_PATH = Path("data/teams.csv")
TEAM_STATES_OUTPUT_PATH = Path("data/team_states.csv")

INITIAL_ELO = 1500.0
K = 20.0
HOME_ADVANTAGE = 100.0
CARRYOVER = 0.75

TEAM_COLUMNS = [
    "game_date",
    "game_id",
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

HOME_STATE_COLUMN_MAP = {
    "home_elo_pre": "elo_pre",
    "home_avg_pts_for_last5": "avg_pts_for_last5",
    "home_avg_pts_against_last5": "avg_pts_against_last5",
    "home_win_pct_last5": "win_pct_last5",
    "home_netrtg_last10": "netrtg_last10",
    "home_efg_last10": "efg_last10",
    "home_tov_pct_last10": "tov_pct_last10",
    "home_orb_pct_last10": "orb_pct_last10",
    "home_ftr_last10": "ftr_last10",
}

AWAY_STATE_COLUMN_MAP = {
    "away_elo_pre": "elo_pre",
    "away_avg_pts_for_last5": "avg_pts_for_last5",
    "away_avg_pts_against_last5": "avg_pts_against_last5",
    "away_win_pct_last5": "win_pct_last5",
    "away_netrtg_last10": "netrtg_last10",
    "away_efg_last10": "efg_last10",
    "away_tov_pct_last10": "tov_pct_last10",
    "away_orb_pct_last10": "orb_pct_last10",
    "away_ftr_last10": "ftr_last10",
}

FINAL_FEATURE_USECOLS = [
    "game_date",
    "game_id",
    "team_id_home",
    "team_id_away",
] + list(HOME_STATE_COLUMN_MAP.keys()) + list(AWAY_STATE_COLUMN_MAP.keys())

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


def _iter_processed_games(chunksize: int = 20000):
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


def build_teams_csv() -> None:
    if not PROCESSED_GAMES_PATH.is_file():
        raise FileNotFoundError(f"Missing source file: {PROCESSED_GAMES_PATH}")

    games = pd.read_csv(PROCESSED_GAMES_PATH, usecols=TEAM_COLUMNS)
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    games["game_id_rank"] = pd.to_numeric(games["game_id"], errors="coerce").fillna(-1)

    home = games[["team_id_home", "team_name_home", "team_abbreviation_home", "game_date", "game_id_rank"]].rename(
        columns={
            "team_id_home": "team_id",
            "team_name_home": "team_name",
            "team_abbreviation_home": "team_abbreviation",
        }
    )
    away = games[["team_id_away", "team_name_away", "team_abbreviation_away", "game_date", "game_id_rank"]].rename(
        columns={
            "team_id_away": "team_id",
            "team_name_away": "team_name",
            "team_abbreviation_away": "team_abbreviation",
        }
    )

    teams = pd.concat([home, away], ignore_index=True)
    teams["team_id"] = teams["team_id"].map(_normalize_team_id)
    teams["team_name"] = teams["team_name"].astype(str).str.strip()
    teams["team_abbreviation"] = teams["team_abbreviation"].astype(str).str.strip().str.upper()
    teams = teams[
        (teams["team_id"] != "")
        & teams["team_name"].ne("")
        & teams["game_date"].notna()
    ]

    teams = teams.sort_values(["team_id", "game_date", "game_id_rank"])
    latest = teams.drop_duplicates(subset=["team_id"], keep="last").copy()
    out = latest[["team_id", "team_name", "team_abbreviation"]].sort_values("team_name")

    TEAMS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(TEAMS_OUTPUT_PATH, index=False)
    print(f"Saved team index: {TEAMS_OUTPUT_PATH} ({len(out)} rows)")


def build_latest_states_from_features(features_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    ordered = features_df.copy()
    ordered["game_date"] = pd.to_datetime(ordered["game_date"], errors="coerce")
    ordered["game_id_rank"] = pd.to_numeric(ordered["game_id"], errors="coerce").fillna(-1)
    ordered["team_id_home"] = ordered["team_id_home"].map(_normalize_team_id)
    ordered["team_id_away"] = ordered["team_id_away"].map(_normalize_team_id)
    ordered = ordered.dropna(subset=["game_date"])
    ordered = ordered.sort_values(["game_date", "game_id_rank"]).reset_index(drop=True)

    home = ordered[
        ["game_date", "game_id_rank", "team_id_home"] + list(HOME_STATE_COLUMN_MAP.keys())
    ].rename(columns={"team_id_home": "team_id", **HOME_STATE_COLUMN_MAP})
    away = ordered[
        ["game_date", "game_id_rank", "team_id_away"] + list(AWAY_STATE_COLUMN_MAP.keys())
    ].rename(columns={"team_id_away": "team_id", **AWAY_STATE_COLUMN_MAP})

    all_states = pd.concat([home, away], ignore_index=True)
    all_states["team_id"] = all_states["team_id"].map(_normalize_team_id)
    all_states = all_states[all_states["team_id"] != ""]
    all_states = all_states.sort_values(["team_id", "game_date", "game_id_rank"])
    latest = all_states.groupby("team_id", sort=False).tail(1).copy()

    states: dict[str, dict[str, Any]] = {}
    for row in latest.to_dict("records"):
        team_id = str(row["team_id"])
        states[team_id] = {
            "elo_pre": _to_float(row.get("elo_pre")),
            "avg_pts_for_last5": _to_float(row.get("avg_pts_for_last5")),
            "avg_pts_against_last5": _to_float(row.get("avg_pts_against_last5")),
            "win_pct_last5": _to_float(row.get("win_pct_last5")),
            "netrtg_last10": _to_float(row.get("netrtg_last10")),
            "efg_last10": _to_float(row.get("efg_last10")),
            "tov_pct_last10": _to_float(row.get("tov_pct_last10")),
            "orb_pct_last10": _to_float(row.get("orb_pct_last10")),
            "ftr_last10": _to_float(row.get("ftr_last10")),
            "last_game_date": pd.Timestamp(row["game_date"]),
        }
    return states


def build_latest_states_from_processed_games() -> dict[str, dict[str, Any]]:
    if not PROCESSED_GAMES_PATH.is_file():
        raise FileNotFoundError(f"Missing source file: {PROCESSED_GAMES_PATH}")

    ratings: dict[str, float] = {}
    trackers: dict[str, dict[str, Any]] = {}
    current_season: float | None = None

    for chunk in _iter_processed_games():
        for _, row in chunk.iterrows():
            season = row.get("season_id")
            if (
                current_season is not None
                and pd.notna(season)
                and float(season) != current_season
            ):
                for team in ratings:
                    ratings[team] = CARRYOVER * ratings[team] + (1.0 - CARRYOVER) * INITIAL_ELO
            if pd.notna(season):
                current_season = float(season)

            home_id = _normalize_team_id(row.get("team_id_home"))
            away_id = _normalize_team_id(row.get("team_id_away"))
            if not home_id or not away_id:
                continue

            trackers.setdefault(home_id, _init_tracker())
            trackers.setdefault(away_id, _init_tracker())
            ratings.setdefault(home_id, INITIAL_ELO)
            ratings.setdefault(away_id, INITIAL_ELO)

            home_elo = ratings[home_id]
            away_elo = ratings[away_id]
            home_win = _parse_home_win(row.get("wl_home"))
            if pd.isna(home_win):
                continue

            pts_home = _to_float(row.get("pts_home"))
            pts_away = _to_float(row.get("pts_away"))
            point_diff = abs(
                (pts_home if pd.notna(pts_home) else 0.0)
                - (pts_away if pd.notna(pts_away) else 0.0)
            )
            expected_home = 1.0 / (1.0 + 10.0 ** ((away_elo - (home_elo + HOME_ADVANTAGE)) / 400.0))
            mov_multiplier = math.log(point_diff + 1.0) * (2.2 / ((home_elo - away_elo) * 0.001 + 2.2))

            ratings[home_id] = home_elo + K * mov_multiplier * (home_win - expected_home)
            ratings[away_id] = away_elo + K * mov_multiplier * ((1.0 - home_win) - (1.0 - expected_home))

            game_date = pd.Timestamp(row["game_date"])
            home_tracker = trackers[home_id]
            away_tracker = trackers[away_id]
            home_tracker["elo"] = ratings[home_id]
            away_tracker["elo"] = ratings[away_id]
            home_tracker["last_game_date"] = game_date
            away_tracker["last_game_date"] = game_date

            home_tracker["pts_for_last5"].append(pts_home)
            home_tracker["pts_against_last5"].append(pts_away)
            home_tracker["win_last5"].append(home_win)
            away_tracker["pts_for_last5"].append(pts_away)
            away_tracker["pts_against_last5"].append(pts_home)
            away_tracker["win_last5"].append(1.0 - home_win)

            possessions = _compute_possessions(
                _to_float(row.get("fga_home")),
                _to_float(row.get("fta_home")),
                _to_float(row.get("oreb_home")),
                _to_float(row.get("tov_home")),
                _to_float(row.get("fga_away")),
                _to_float(row.get("fta_away")),
                _to_float(row.get("oreb_away")),
                _to_float(row.get("tov_away")),
            )
            home_tracker["pf_last10"].append(pts_home)
            home_tracker["pa_last10"].append(pts_away)
            home_tracker["poss_last10"].append(possessions)
            away_tracker["pf_last10"].append(pts_away)
            away_tracker["pa_last10"].append(pts_home)
            away_tracker["poss_last10"].append(possessions)

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

    states: dict[str, dict[str, Any]] = {}
    for team_id, tracker in trackers.items():
        if tracker["last_game_date"] is None:
            continue
        states[team_id] = _tracker_to_state(tracker)
    return states


def build_team_states_csv() -> None:
    if FINAL_FEATURES_PATH.is_file():
        features_df = pd.read_csv(FINAL_FEATURES_PATH, usecols=FINAL_FEATURE_USECOLS)
        states = build_latest_states_from_features(features_df)
        print(f"Built team states from {FINAL_FEATURES_PATH}")
    else:
        states = build_latest_states_from_processed_games()
        print(f"Built team states from {PROCESSED_GAMES_PATH}")

    if not states:
        raise RuntimeError("No matchup states available to persist.")

    rows = []
    for team_id, state in states.items():
        row = {
            "team_id": str(team_id),
            "last_game_date": pd.Timestamp(state["last_game_date"]),
        }
        for column in STATE_COLUMNS:
            row[column] = _to_float(state.get(column))
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("team_id").reset_index(drop=True)
    TEAM_STATES_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(TEAM_STATES_OUTPUT_PATH, index=False)
    print(f"Saved team states: {TEAM_STATES_OUTPUT_PATH} ({len(out)} rows)")


def main() -> None:
    build_teams_csv()
    build_team_states_csv()


if __name__ == "__main__":
    main()
