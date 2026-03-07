from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import duckdb
import joblib
import pandas as pd

FEATURES = [
    "home_elo_pre",
    "away_elo_pre",
    "home_avg_pts_for_last5",
    "home_avg_pts_against_last5",
    "away_avg_pts_for_last5",
    "away_avg_pts_against_last5",
    "home_win_pct_last5",
    "away_win_pct_last5",
    "home_rest_days",
    "away_rest_days",
    "home_b2b",
    "away_b2b",
    "home_netrtg_last10",
    "away_netrtg_last10",
    "home_efg_last10",
    "home_tov_pct_last10",
    "home_orb_pct_last10",
    "home_ftr_last10",
    "away_efg_last10",
    "away_tov_pct_last10",
    "away_orb_pct_last10",
    "away_ftr_last10",
    "net_diff_last5",
    "win_pct_diff_last5",
    "elo_diff",
    "rest_diff",
    "b2b_diff",
    "netrtg_diff_last10",
    "efg_diff_last10",
    "tov_pct_diff_last10",
    "orb_pct_diff_last10",
    "ftr_diff_last10",
]


def parse_args() -> Any:
    parser = ArgumentParser(description="Predict home win probability for an NBA matchup.")
    parser.add_argument("--home", required=True, help="Home team abbreviation or full name (e.g. IND)")
    parser.add_argument("--away", required=True, help="Away team abbreviation or full name (e.g. LAL)")
    parser.add_argument(
        "--game-date",
        default=None,
        help="Game date in YYYY-MM-DD format. Defaults to one day after the later last-game date.",
    )
    return parser.parse_args()


def normalize(s: str) -> str:
    return s.strip().lower()


def resolve_team_id(conn: duckdb.DuckDBPyConnection, team_query: str) -> str:
    teams = conn.execute(
        """
        SELECT
            id,
            full_name,
            abbreviation,
            nickname,
            city
        FROM team
        """
    ).df()

    q = normalize(team_query)
    exact = teams[
        (teams["id"].str.lower() == q)
        | (teams["abbreviation"].str.lower() == q)
        | (teams["full_name"].str.lower() == q)
    ]
    if len(exact) == 1:
        return exact.iloc[0]["id"]

    partial = teams[
        teams["full_name"].str.lower().str.contains(q, regex=False)
        | teams["nickname"].str.lower().str.contains(q, regex=False)
        | teams["city"].str.lower().str.contains(q, regex=False)
    ]
    if len(partial) == 1:
        return partial.iloc[0]["id"]

    if len(exact) > 1:
        matches = ", ".join(exact["full_name"].tolist())
        raise ValueError(f"Ambiguous team '{team_query}'. Matches: {matches}")
    if len(partial) > 1:
        matches = ", ".join(partial["full_name"].tolist())
        raise ValueError(f"Ambiguous team '{team_query}'. Matches: {matches}")

    raise ValueError(f"Team '{team_query}' not found.")


def latest_team_state(conn: duckdb.DuckDBPyConnection, team_id: str) -> dict[str, Any]:
    row = conn.execute(
        """
        WITH team_states AS (
            SELECT
                game_date,
                game_id,
                team_id_home AS team_id,
                home_elo_pre AS elo_pre,
                home_avg_pts_for_last5 AS avg_pts_for_last5,
                home_avg_pts_against_last5 AS avg_pts_against_last5,
                home_win_pct_last5 AS win_pct_last5,
                home_netrtg_last10 AS netrtg_last10,
                home_efg_last10 AS efg_last10,
                home_tov_pct_last10 AS tov_pct_last10,
                home_orb_pct_last10 AS orb_pct_last10,
                home_ftr_last10 AS ftr_last10
            FROM final_features
            UNION ALL
            SELECT
                game_date,
                game_id,
                team_id_away AS team_id,
                away_elo_pre AS elo_pre,
                away_avg_pts_for_last5 AS avg_pts_for_last5,
                away_avg_pts_against_last5 AS avg_pts_against_last5,
                away_win_pct_last5 AS win_pct_last5,
                away_netrtg_last10 AS netrtg_last10,
                away_efg_last10 AS efg_last10,
                away_tov_pct_last10 AS tov_pct_last10,
                away_orb_pct_last10 AS orb_pct_last10,
                away_ftr_last10 AS ftr_last10
            FROM final_features
        )
        SELECT *
        FROM team_states
        WHERE team_id = ?
        ORDER BY game_date DESC, game_id DESC
        LIMIT 1
        """,
        [team_id],
    ).fetchone()

    if row is None:
        raise ValueError(f"No historical features found for team id '{team_id}'.")

    columns = [
        "game_date",
        "game_id",
        "team_id",
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
    return dict(zip(columns, row))


def last_game_date(conn: duckdb.DuckDBPyConnection, team_id: str) -> pd.Timestamp:
    row = conn.execute(
        """
        SELECT MAX(game_date) AS last_game_date
        FROM team_game_long
        WHERE team_id = ?
        """,
        [team_id],
    ).fetchone()
    if row is None or row[0] is None:
        raise ValueError(f"No game history found for team id '{team_id}'.")
    return pd.Timestamp(row[0])


def build_feature_row(
    home_state: dict[str, Any],
    away_state: dict[str, Any],
    home_rest_days: int,
    away_rest_days: int,
) -> pd.DataFrame:
    home_b2b = int(home_rest_days <= 1)
    away_b2b = int(away_rest_days <= 1)
    home_net_last5 = home_state["avg_pts_for_last5"] - home_state["avg_pts_against_last5"]
    away_net_last5 = away_state["avg_pts_for_last5"] - away_state["avg_pts_against_last5"]

    row = {
        "home_elo_pre": home_state["elo_pre"],
        "away_elo_pre": away_state["elo_pre"],
        "home_avg_pts_for_last5": home_state["avg_pts_for_last5"],
        "home_avg_pts_against_last5": home_state["avg_pts_against_last5"],
        "away_avg_pts_for_last5": away_state["avg_pts_for_last5"],
        "away_avg_pts_against_last5": away_state["avg_pts_against_last5"],
        "home_win_pct_last5": home_state["win_pct_last5"],
        "away_win_pct_last5": away_state["win_pct_last5"],
        "home_rest_days": home_rest_days,
        "away_rest_days": away_rest_days,
        "home_b2b": home_b2b,
        "away_b2b": away_b2b,
        "home_netrtg_last10": home_state["netrtg_last10"],
        "away_netrtg_last10": away_state["netrtg_last10"],
        "home_efg_last10": home_state["efg_last10"],
        "home_tov_pct_last10": home_state["tov_pct_last10"],
        "home_orb_pct_last10": home_state["orb_pct_last10"],
        "home_ftr_last10": home_state["ftr_last10"],
        "away_efg_last10": away_state["efg_last10"],
        "away_tov_pct_last10": away_state["tov_pct_last10"],
        "away_orb_pct_last10": away_state["orb_pct_last10"],
        "away_ftr_last10": away_state["ftr_last10"],
        "net_diff_last5": home_net_last5 - away_net_last5,
        "win_pct_diff_last5": home_state["win_pct_last5"] - away_state["win_pct_last5"],
        "elo_diff": home_state["elo_pre"] - away_state["elo_pre"],
        "rest_diff": home_rest_days - away_rest_days,
        "b2b_diff": home_b2b - away_b2b,
        "netrtg_diff_last10": home_state["netrtg_last10"] - away_state["netrtg_last10"],
        "efg_diff_last10": home_state["efg_last10"] - away_state["efg_last10"],
        "tov_pct_diff_last10": home_state["tov_pct_last10"] - away_state["tov_pct_last10"],
        "orb_pct_diff_last10": home_state["orb_pct_last10"] - away_state["orb_pct_last10"],
        "ftr_diff_last10": home_state["ftr_last10"] - away_state["ftr_last10"],
    }
    return pd.DataFrame([row], columns=FEATURES)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    db_path = repo_root / "data" / "nba.duckdb"
    model_path = repo_root / "models" / "logistic_model.pkl"

    conn = duckdb.connect(str(db_path))
    model = joblib.load(model_path)

    home_id = resolve_team_id(conn, args.home)
    away_id = resolve_team_id(conn, args.away)
    if home_id == away_id:
        raise ValueError("Home and away teams must be different.")

    teams = conn.execute(
        "SELECT id, full_name, abbreviation FROM team WHERE id IN (?, ?)",
        [home_id, away_id],
    ).df()
    home_meta = teams[teams["id"] == home_id].iloc[0]
    away_meta = teams[teams["id"] == away_id].iloc[0]

    home_state = latest_team_state(conn, home_id)
    away_state = latest_team_state(conn, away_id)
    home_last_game = last_game_date(conn, home_id)
    away_last_game = last_game_date(conn, away_id)

    if args.game_date:
        game_date = pd.Timestamp(args.game_date)
    else:
        game_date = max(home_last_game, away_last_game) + pd.Timedelta(days=1)

    home_rest_days = max((game_date - home_last_game).days, 0)
    away_rest_days = max((game_date - away_last_game).days, 0)

    x = build_feature_row(home_state, away_state, home_rest_days, away_rest_days)
    home_win_prob = float(model.predict_proba(x)[:, 1][0])
    away_win_prob = 1.0 - home_win_prob
    predicted_winner = home_meta["full_name"] if home_win_prob >= 0.5 else away_meta["full_name"]

    print(
        f"Matchup: {away_meta['full_name']} ({away_meta['abbreviation']}) at "
        f"{home_meta['full_name']} ({home_meta['abbreviation']})"
    )
    print(f"Assumed game date: {game_date.date()}")
    print(f"Home rest days: {home_rest_days} | Away rest days: {away_rest_days}")
    print(f"Home win probability: {home_win_prob:.4f}")
    print(f"Away win probability: {away_win_prob:.4f}")
    print(f"Predicted winner: {predicted_winner}")


if __name__ == "__main__":
    main()
