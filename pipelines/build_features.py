from __future__ import annotations

from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd

REQUIRED_MODEL_FEATURES = [
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

IDENTITY_COLUMNS = [
    "game_id",
    "game_date",
    "season_id",
    "team_id_home",
    "team_id_away",
    "home_win",
]

INITIAL_ELO = 1500.0
K = 20.0
HOME_ADVANTAGE = 100.0
CARRYOVER = 0.75


def table_exists(conn: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    row = conn.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = 'main' AND table_name = ?
        """,
        [table_name],
    ).fetchone()
    return bool(row and row[0] > 0)


def get_columns(conn: duckdb.DuckDBPyConnection, table_name: str) -> list[str]:
    if not table_exists(conn, table_name):
        return []
    rows = conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
    return [str(row[1]) for row in rows]


def print_available_columns(
    conn: duckdb.DuckDBPyConnection, tables: Iterable[str]
) -> None:
    for table in tables:
        cols = get_columns(conn, table)
        if cols:
            print(f"[DEBUG] available columns in '{table}': {cols}")
        else:
            print(f"[DEBUG] table '{table}' not found.")


def safe_query(
    conn: duckdb.DuckDBPyConnection,
    query: str,
    *,
    label: str,
    tables_to_debug: Iterable[str],
) -> pd.DataFrame:
    try:
        return conn.execute(query).df()
    except Exception as exc:
        print(f"[ERROR] query failed at: {label}")
        print_available_columns(conn, tables_to_debug)
        raise RuntimeError(f"Failed query at '{label}': {exc}") from exc


def first_existing(columns: set[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def build_model_base_enriched(conn: duckdb.DuckDBPyConnection) -> str:
    source_table = "model_base" if table_exists(conn, "model_base") else "game"
    if not table_exists(conn, source_table):
        raise RuntimeError("Neither 'model_base' nor 'game' table exists in DuckDB.")

    cols = set(get_columns(conn, source_table))

    game_id_col = first_existing(cols, ["game_id", "id"])
    game_date_col = first_existing(
        cols, ["game_date", "game_date_est", "date_game", "date"]
    )
    season_col = first_existing(cols, ["season_id", "season", "season_year"])
    home_id_col = first_existing(
        cols,
        ["team_id_home", "home_team_id", "team_home_id", "team_home"],
    )
    away_id_col = first_existing(
        cols,
        ["team_id_away", "away_team_id", "team_away_id", "team_away"],
    )
    home_name_col = first_existing(
        cols,
        ["team_name_home", "home_team_name", "home_team", "team_home_name"],
    )
    away_name_col = first_existing(
        cols,
        ["team_name_away", "away_team_name", "away_team", "team_away_name"],
    )

    if game_date_col is None:
        raise RuntimeError(
            f"Missing game date column in '{source_table}'. "
            "Expected one of: game_date/game_date_est/date_game/date"
        )
    if game_id_col is None:
        game_id_expr = "ROW_NUMBER() OVER (ORDER BY game_date_raw, team_id_home_raw, team_id_away_raw)"
    else:
        game_id_expr = f"CAST({game_id_col} AS VARCHAR)"

    team_home_raw_expr = (
        f"CAST({home_id_col} AS VARCHAR)"
        if home_id_col is not None
        else (
            f"CAST({home_name_col} AS VARCHAR)" if home_name_col is not None else "NULL"
        )
    )
    team_away_raw_expr = (
        f"CAST({away_id_col} AS VARCHAR)"
        if away_id_col is not None
        else (
            f"CAST({away_name_col} AS VARCHAR)" if away_name_col is not None else "NULL"
        )
    )

    season_expr = (
        f"TRY_CAST({season_col} AS INTEGER)"
        if season_col is not None
        else "EXTRACT(YEAR FROM CAST(game_date_raw AS DATE))"
    )
    wl_home_col = first_existing(cols, ["wl_home", "home_wl", "wl"])
    wl_away_col = first_existing(cols, ["wl_away", "away_wl"])
    home_win_col = first_existing(cols, ["home_win"])
    away_win_col = first_existing(cols, ["away_win"])

    if home_win_col is not None:
        home_win_expr = f"""
            CASE
                WHEN TRY_CAST({home_win_col} AS DOUBLE) IS NOT NULL
                    THEN CASE WHEN TRY_CAST({home_win_col} AS DOUBLE) >= 0.5 THEN 1 ELSE 0 END
                WHEN UPPER(TRIM(CAST({home_win_col} AS VARCHAR))) IN ('W', 'WIN', 'TRUE', 'T') THEN 1
                WHEN UPPER(TRIM(CAST({home_win_col} AS VARCHAR))) IN ('L', 'LOSS', 'FALSE', 'F') THEN 0
                ELSE NULL
            END
        """
    elif wl_home_col is not None:
        home_win_expr = f"""
            CASE
                WHEN UPPER(TRIM(COALESCE(CAST({wl_home_col} AS VARCHAR), ''))) = 'W' THEN 1
                WHEN UPPER(TRIM(COALESCE(CAST({wl_home_col} AS VARCHAR), ''))) = 'L' THEN 0
                ELSE NULL
            END
        """
    elif away_win_col is not None:
        home_win_expr = f"""
            CASE
                WHEN TRY_CAST({away_win_col} AS DOUBLE) IS NOT NULL
                    THEN CASE WHEN TRY_CAST({away_win_col} AS DOUBLE) >= 0.5 THEN 0 ELSE 1 END
                ELSE NULL
            END
        """
    elif wl_away_col is not None:
        home_win_expr = f"""
            CASE
                WHEN UPPER(TRIM(COALESCE(CAST({wl_away_col} AS VARCHAR), ''))) = 'W' THEN 0
                WHEN UPPER(TRIM(COALESCE(CAST({wl_away_col} AS VARCHAR), ''))) = 'L' THEN 1
                ELSE NULL
            END
        """
    else:
        home_win_expr = "NULL"

    away_win_expr = f"""
        CASE
            WHEN ({home_win_expr}) IS NULL THEN NULL
            WHEN ({home_win_expr}) = 1 THEN 0
            ELSE 1
        END
    """

    pts_home_col = first_existing(cols, ["pts_home", "home_pts"])
    pts_away_col = first_existing(cols, ["pts_away", "away_pts"])

    stat_candidates = {
        "fgm_home": ["fgm_home"],
        "fg3m_home": ["fg3m_home"],
        "fga_home": ["fga_home"],
        "fta_home": ["fta_home"],
        "oreb_home": ["oreb_home"],
        "dreb_home": ["dreb_home"],
        "tov_home": ["tov_home"],
        "fgm_away": ["fgm_away"],
        "fg3m_away": ["fg3m_away"],
        "fga_away": ["fga_away"],
        "fta_away": ["fta_away"],
        "oreb_away": ["oreb_away"],
        "dreb_away": ["dreb_away"],
        "tov_away": ["tov_away"],
    }

    stat_exprs: list[str] = []
    for out_col, cands in stat_candidates.items():
        found = first_existing(cols, cands)
        if found is None:
            stat_exprs.append(f"CAST(NULL AS DOUBLE) AS {out_col}")
        else:
            stat_exprs.append(f"TRY_CAST({found} AS DOUBLE) AS {out_col}")

    season_source_expr = (
        f"CAST({season_col} AS VARCHAR)" if season_col is not None else "NULL"
    )

    query = f"""
        CREATE OR REPLACE TABLE model_base_enriched AS
        WITH src AS (
            SELECT
                {game_id_expr} AS game_id,
                CAST({game_date_col} AS DATE) AS game_date_raw,
                {season_expr} AS season_id,
                {season_source_expr} AS season_source,
                {team_home_raw_expr} AS team_id_home_raw,
                {team_away_raw_expr} AS team_id_away_raw,
                {home_win_expr} AS home_win_raw,
                {away_win_expr} AS away_win_raw,
                {f"TRY_CAST({pts_home_col} AS DOUBLE)" if pts_home_col is not None else "CAST(NULL AS DOUBLE)"} AS pts_home,
                {f"TRY_CAST({pts_away_col} AS DOUBLE)" if pts_away_col is not None else "CAST(NULL AS DOUBLE)"} AS pts_away,
                {", ".join(stat_exprs)}
            FROM {source_table}
        ),
        normalized AS (
            SELECT
                CAST(game_id AS VARCHAR) AS game_id,
                CAST(game_date_raw AS DATE) AS game_date,
                season_id,
                season_source,
                team_id_home_raw,
                team_id_away_raw,
                home_win_raw AS home_win,
                away_win_raw AS away_win,
                pts_home,
                pts_away,
                fgm_home,
                fg3m_home,
                fga_home,
                fta_home,
                oreb_home,
                dreb_home,
                tov_home,
                fgm_away,
                fg3m_away,
                fga_away,
                fta_away,
                oreb_away,
                dreb_away,
                tov_away
            FROM src
        ),
        teams AS (
            SELECT team_id_home_raw AS team_key FROM normalized
            UNION
            SELECT team_id_away_raw AS team_key FROM normalized
        ),
        team_ref AS (
            SELECT
                team_key,
                ROW_NUMBER() OVER (ORDER BY team_key) AS team_num
            FROM teams
            WHERE team_key IS NOT NULL
        )
        SELECT
            n.game_id,
            n.game_date,
            COALESCE(
                n.season_id,
                TRY_CAST(SUBSTR(n.season_source, 1, 4) AS INTEGER),
                EXTRACT(YEAR FROM n.game_date)
            ) AS season_id,
            COALESCE(n.team_id_home_raw, CAST(th.team_num AS VARCHAR)) AS team_id_home,
            COALESCE(n.team_id_away_raw, CAST(ta.team_num AS VARCHAR)) AS team_id_away,
            CASE WHEN n.home_win IN (0, 1) THEN CAST(n.home_win AS INTEGER) ELSE NULL END AS home_win,
            CASE WHEN n.away_win IN (0, 1) THEN CAST(n.away_win AS INTEGER) ELSE NULL END AS away_win,
            n.pts_home,
            n.pts_away,
            n.fgm_home,
            n.fg3m_home,
            n.fga_home,
            n.fta_home,
            n.oreb_home,
            n.dreb_home,
            n.tov_home,
            n.fgm_away,
            n.fg3m_away,
            n.fga_away,
            n.fta_away,
            n.oreb_away,
            n.dreb_away,
            n.tov_away
        FROM normalized n
        LEFT JOIN team_ref th ON n.team_id_home_raw = th.team_key
        LEFT JOIN team_ref ta ON n.team_id_away_raw = ta.team_key
        ORDER BY n.game_date, n.game_id
    """

    conn.execute(query)
    return "model_base_enriched"


def load_or_build_base_rolling_features(
    conn: duckdb.DuckDBPyConnection, base_table: str
) -> pd.DataFrame:
    expected_roll_cols = [
        "game_id",
        "game_date",
        "season_id",
        "team_id_home",
        "team_id_away",
        "home_win",
        "home_avg_pts_for_last5",
        "home_avg_pts_against_last5",
        "home_win_pct_last5",
        "away_avg_pts_for_last5",
        "away_avg_pts_against_last5",
        "away_win_pct_last5",
    ]

    if table_exists(conn, "game_features_clean"):
        available = set(get_columns(conn, "game_features_clean"))
        if all(col in available for col in expected_roll_cols):
            print("Loading rolling feature data from game_features_clean...")
            return safe_query(
                conn,
                """
                SELECT
                    CAST(game_id AS VARCHAR) AS game_id,
                    CAST(game_date AS DATE) AS game_date,
                    COALESCE(TRY_CAST(season_id AS INTEGER), EXTRACT(YEAR FROM CAST(game_date AS DATE))) AS season_id,
                    CAST(team_id_home AS VARCHAR) AS team_id_home,
                    CAST(team_id_away AS VARCHAR) AS team_id_away,
                    CASE
                        WHEN TRY_CAST(home_win AS DOUBLE) >= 0.5 THEN 1
                        WHEN TRY_CAST(home_win AS DOUBLE) < 0.5 THEN 0
                        ELSE NULL
                    END AS home_win,
                    TRY_CAST(home_avg_pts_for_last5 AS DOUBLE) AS home_avg_pts_for_last5,
                    TRY_CAST(home_avg_pts_against_last5 AS DOUBLE) AS home_avg_pts_against_last5,
                    TRY_CAST(home_win_pct_last5 AS DOUBLE) AS home_win_pct_last5,
                    TRY_CAST(away_avg_pts_for_last5 AS DOUBLE) AS away_avg_pts_for_last5,
                    TRY_CAST(away_avg_pts_against_last5 AS DOUBLE) AS away_avg_pts_against_last5,
                    TRY_CAST(away_win_pct_last5 AS DOUBLE) AS away_win_pct_last5
                FROM game_features_clean
                ORDER BY game_date, game_id
                """,
                label="load game_features_clean",
                tables_to_debug=["game_features_clean"],
            )

    print("Building rolling feature data from base game history...")
    base_games = safe_query(
        conn,
        f"""
        SELECT
            CAST(game_id AS VARCHAR) AS game_id,
            CAST(game_date AS DATE) AS game_date,
            COALESCE(TRY_CAST(season_id AS INTEGER), EXTRACT(YEAR FROM CAST(game_date AS DATE))) AS season_id,
            CAST(team_id_home AS VARCHAR) AS team_id_home,
            CAST(team_id_away AS VARCHAR) AS team_id_away,
            CASE
                WHEN TRY_CAST(home_win AS DOUBLE) >= 0.5 THEN 1
                WHEN TRY_CAST(home_win AS DOUBLE) < 0.5 THEN 0
                ELSE NULL
            END AS home_win,
            TRY_CAST(away_win AS DOUBLE) AS away_win,
            TRY_CAST(pts_home AS DOUBLE) AS pts_home,
            TRY_CAST(pts_away AS DOUBLE) AS pts_away
        FROM {base_table}
        ORDER BY game_date, game_id
        """,
        label="load base games",
        tables_to_debug=[base_table],
    )

    if table_exists(conn, "team_game_long"):
        long_cols = set(get_columns(conn, "team_game_long"))
        required_long_cols = {
            "game_id",
            "game_date",
            "season_id",
            "team_id",
            "is_home",
            "points_for",
            "points_against",
            "win",
        }
        if required_long_cols.issubset(long_cols):
            team_long = safe_query(
                conn,
                """
                SELECT
                    CAST(game_id AS VARCHAR) AS game_id,
                    CAST(game_date AS DATE) AS game_date,
                    COALESCE(TRY_CAST(season_id AS INTEGER), EXTRACT(YEAR FROM CAST(game_date AS DATE))) AS season_id,
                    CAST(team_id AS VARCHAR) AS team_id,
                    CASE
                        WHEN TRY_CAST(is_home AS DOUBLE) >= 0.5 THEN 1
                        ELSE 0
                    END AS is_home,
                    TRY_CAST(points_for AS DOUBLE) AS points_for,
                    TRY_CAST(points_against AS DOUBLE) AS points_against,
                    CASE
                        WHEN TRY_CAST(win AS DOUBLE) >= 0.5 THEN 1
                        WHEN TRY_CAST(win AS DOUBLE) < 0.5 THEN 0
                        ELSE NULL
                    END AS win
                FROM team_game_long
                ORDER BY team_id, game_date, game_id
                """,
                label="load team_game_long",
                tables_to_debug=["team_game_long"],
            )
        else:
            team_long = pd.DataFrame()
    else:
        team_long = pd.DataFrame()

    if team_long.empty:
        long_home = base_games[
            [
                "game_id",
                "game_date",
                "season_id",
                "team_id_home",
                "pts_home",
                "pts_away",
                "home_win",
            ]
        ].rename(
            columns={
                "team_id_home": "team_id",
                "pts_home": "points_for",
                "pts_away": "points_against",
                "home_win": "win",
            }
        )
        long_home["is_home"] = 1

        long_away = base_games[
            [
                "game_id",
                "game_date",
                "season_id",
                "team_id_away",
                "pts_away",
                "pts_home",
                "away_win",
            ]
        ].rename(
            columns={
                "team_id_away": "team_id",
                "pts_away": "points_for",
                "pts_home": "points_against",
                "away_win": "win",
            }
        )
        long_away["is_home"] = 0
        team_long = pd.concat([long_home, long_away], ignore_index=True)

    team_long = team_long.sort_values(["team_id", "game_date", "game_id"]).copy()
    grouped = team_long.groupby("team_id")

    team_long["avg_pts_for_last5"] = grouped["points_for"].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=3).mean()
    )
    team_long["avg_pts_against_last5"] = grouped["points_against"].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=3).mean()
    )
    team_long["win_pct_last5"] = grouped["win"].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=3).mean()
    )

    home_roll = team_long[team_long["is_home"] == 1][
        [
            "game_id",
            "team_id",
            "avg_pts_for_last5",
            "avg_pts_against_last5",
            "win_pct_last5",
        ]
    ].rename(
        columns={
            "team_id": "team_id_home",
            "avg_pts_for_last5": "home_avg_pts_for_last5",
            "avg_pts_against_last5": "home_avg_pts_against_last5",
            "win_pct_last5": "home_win_pct_last5",
        }
    )

    away_roll = team_long[team_long["is_home"] == 0][
        [
            "game_id",
            "team_id",
            "avg_pts_for_last5",
            "avg_pts_against_last5",
            "win_pct_last5",
        ]
    ].rename(
        columns={
            "team_id": "team_id_away",
            "avg_pts_for_last5": "away_avg_pts_for_last5",
            "avg_pts_against_last5": "away_avg_pts_against_last5",
            "win_pct_last5": "away_win_pct_last5",
        }
    )

    df = base_games[
        [
            "game_id",
            "game_date",
            "season_id",
            "team_id_home",
            "team_id_away",
            "home_win",
        ]
    ].copy()

    df = df.merge(home_roll, on=["game_id", "team_id_home"], how="left")
    df = df.merge(away_roll, on=["game_id", "team_id_away"], how="left")
    return df


def add_elo_features(df: pd.DataFrame, base_games: pd.DataFrame) -> pd.DataFrame:
    print("Computing MOV-adjusted Elo ratings with offseason regression...")

    elo_ratings: dict[str, float] = {}
    home_elo_pre: list[float] = []
    away_elo_pre: list[float] = []
    current_season = None

    games = base_games.sort_values(["game_date", "game_id"]).copy()
    games["season_id"] = pd.to_numeric(games["season_id"], errors="coerce")
    games["home_win"] = pd.to_numeric(games["home_win"], errors="coerce")
    games["pts_home"] = pd.to_numeric(games["pts_home"], errors="coerce")
    games["pts_away"] = pd.to_numeric(games["pts_away"], errors="coerce")

    for _, row in games.iterrows():
        season = row["season_id"]
        if current_season is not None and pd.notna(season) and season != current_season:
            for team in elo_ratings:
                elo_ratings[team] = (
                    CARRYOVER * elo_ratings[team] + (1 - CARRYOVER) * INITIAL_ELO
                )
        if pd.notna(season):
            current_season = season

        home = str(row["team_id_home"])
        away = str(row["team_id_away"])
        elo_ratings.setdefault(home, INITIAL_ELO)
        elo_ratings.setdefault(away, INITIAL_ELO)

        r_home = elo_ratings[home]
        r_away = elo_ratings[away]
        home_elo_pre.append(r_home)
        away_elo_pre.append(r_away)

        expected_home = 1.0 / (
            1.0 + 10.0 ** ((r_away - (r_home + HOME_ADVANTAGE)) / 400.0)
        )
        actual_home = row["home_win"]
        if pd.isna(actual_home):
            actual_home = 0.5

        pts_home = row["pts_home"] if pd.notna(row["pts_home"]) else 0.0
        pts_away = row["pts_away"] if pd.notna(row["pts_away"]) else 0.0
        point_diff = abs(float(pts_home) - float(pts_away))
        mov_multiplier = np.log(point_diff + 1.0) * (
            2.2 / ((r_home - r_away) * 0.001 + 2.2)
        )

        elo_ratings[home] = r_home + K * mov_multiplier * (
            float(actual_home) - expected_home
        )
        elo_ratings[away] = r_away + K * mov_multiplier * (
            (1.0 - float(actual_home)) - (1.0 - expected_home)
        )

    games["home_elo_pre"] = home_elo_pre
    games["away_elo_pre"] = away_elo_pre
    games["elo_diff"] = games["home_elo_pre"] - games["away_elo_pre"]

    return df.merge(
        games[["game_id", "home_elo_pre", "away_elo_pre", "elo_diff"]],
        on="game_id",
        how="left",
    )


def add_rest_day_features(
    conn: duckdb.DuckDBPyConnection, df: pd.DataFrame, base_table: str
) -> pd.DataFrame:
    print("Computing rest day features...")

    if table_exists(conn, "team_game_long"):
        cols = set(get_columns(conn, "team_game_long"))
        required = {"game_id", "game_date", "team_id"}
        if required.issubset(cols):
            team_games = safe_query(
                conn,
                """
                SELECT
                    CAST(game_id AS VARCHAR) AS game_id,
                    CAST(game_date AS DATE) AS game_date,
                    CAST(team_id AS VARCHAR) AS team_id
                FROM team_game_long
                ORDER BY team_id, game_date, game_id
                """,
                label="load rest-day team_game_long",
                tables_to_debug=["team_game_long"],
            )
        else:
            team_games = pd.DataFrame()
    else:
        team_games = pd.DataFrame()

    if team_games.empty:
        base_games = safe_query(
            conn,
            f"""
            SELECT
                CAST(game_id AS VARCHAR) AS game_id,
                CAST(game_date AS DATE) AS game_date,
                CAST(team_id_home AS VARCHAR) AS team_id_home,
                CAST(team_id_away AS VARCHAR) AS team_id_away
            FROM {base_table}
            ORDER BY game_date, game_id
            """,
            label="load base for rest-day fallback",
            tables_to_debug=[base_table],
        )
        home_long = base_games[["game_id", "game_date", "team_id_home"]].rename(
            columns={"team_id_home": "team_id"}
        )
        away_long = base_games[["game_id", "game_date", "team_id_away"]].rename(
            columns={"team_id_away": "team_id"}
        )
        team_games = pd.concat([home_long, away_long], ignore_index=True)
        team_games = team_games.sort_values(["team_id", "game_date", "game_id"])

    team_games["prev_game_date"] = team_games.groupby("team_id")["game_date"].shift(1)
    team_games["rest_days"] = (
        pd.to_datetime(team_games["game_date"])
        - pd.to_datetime(team_games["prev_game_date"])
    ).dt.days

    home_rest = team_games[["game_id", "team_id", "rest_days"]].rename(
        columns={"team_id": "team_id_home", "rest_days": "home_rest_days"}
    )
    away_rest = team_games[["game_id", "team_id", "rest_days"]].rename(
        columns={"team_id": "team_id_away", "rest_days": "away_rest_days"}
    )

    df = df.merge(home_rest, on=["game_id", "team_id_home"], how="left")
    df = df.merge(away_rest, on=["game_id", "team_id_away"], how="left")
    df["rest_diff"] = df["home_rest_days"] - df["away_rest_days"]
    df["home_b2b"] = (pd.to_numeric(df["home_rest_days"], errors="coerce") <= 1).astype(
        float
    )
    df["away_b2b"] = (pd.to_numeric(df["away_rest_days"], errors="coerce") <= 1).astype(
        float
    )
    df["b2b_diff"] = df["home_b2b"] - df["away_b2b"]
    return df


def add_pace_features(
    conn: duckdb.DuckDBPyConnection, df: pd.DataFrame, base_table: str
) -> pd.DataFrame:
    print("Computing pace-adjusted efficiency features...")
    required_cols = {
        "game_id",
        "game_date",
        "team_id_home",
        "team_id_away",
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
    }

    available_cols = set(get_columns(conn, base_table))
    if not required_cols.issubset(available_cols):
        print(
            "[WARN] Skipping pace features due to missing columns. "
            f"Missing: {sorted(required_cols - available_cols)}"
        )
        df["home_netrtg_last10"] = np.nan
        df["away_netrtg_last10"] = np.nan
        df["netrtg_diff_last10"] = np.nan
        return df

    pace_games = safe_query(
        conn,
        f"""
        SELECT
            CAST(game_id AS VARCHAR) AS game_id,
            CAST(game_date AS DATE) AS game_date,
            CAST(team_id_home AS VARCHAR) AS team_id_home,
            CAST(team_id_away AS VARCHAR) AS team_id_away,
            TRY_CAST(pts_home AS DOUBLE) AS pts_home,
            TRY_CAST(pts_away AS DOUBLE) AS pts_away,
            TRY_CAST(fga_home AS DOUBLE) AS fga_home,
            TRY_CAST(fta_home AS DOUBLE) AS fta_home,
            TRY_CAST(oreb_home AS DOUBLE) AS oreb_home,
            TRY_CAST(tov_home AS DOUBLE) AS tov_home,
            TRY_CAST(fga_away AS DOUBLE) AS fga_away,
            TRY_CAST(fta_away AS DOUBLE) AS fta_away,
            TRY_CAST(oreb_away AS DOUBLE) AS oreb_away,
            TRY_CAST(tov_away AS DOUBLE) AS tov_away
        FROM {base_table}
        ORDER BY game_date, game_id
        """,
        label="load pace source data",
        tables_to_debug=[base_table],
    )

    pace_games["possessions"] = 0.5 * (
        (
            pace_games["fga_home"]
            + 0.44 * pace_games["fta_home"]
            - pace_games["oreb_home"]
            + pace_games["tov_home"]
        )
        + (
            pace_games["fga_away"]
            + 0.44 * pace_games["fta_away"]
            - pace_games["oreb_away"]
            + pace_games["tov_away"]
        )
    )

    home_eff_long = pace_games[
        ["game_id", "game_date", "team_id_home", "pts_home", "pts_away", "possessions"]
    ].rename(
        columns={
            "team_id_home": "team_id",
            "pts_home": "points_for",
            "pts_away": "points_against",
        }
    )
    away_eff_long = pace_games[
        ["game_id", "game_date", "team_id_away", "pts_away", "pts_home", "possessions"]
    ].rename(
        columns={
            "team_id_away": "team_id",
            "pts_away": "points_for",
            "pts_home": "points_against",
        }
    )

    team_eff = pd.concat([home_eff_long, away_eff_long], ignore_index=True)
    team_eff = team_eff.sort_values(["team_id", "game_date", "game_id"])
    grouped = team_eff.groupby("team_id")
    team_eff["pf_last10"] = grouped["points_for"].transform(
        lambda s: s.shift(1).rolling(window=10, min_periods=5).sum()
    )
    team_eff["pa_last10"] = grouped["points_against"].transform(
        lambda s: s.shift(1).rolling(window=10, min_periods=5).sum()
    )
    team_eff["poss_last10"] = grouped["possessions"].transform(
        lambda s: s.shift(1).rolling(window=10, min_periods=5).sum()
    )

    team_eff["ortg_last10"] = 100.0 * team_eff["pf_last10"] / team_eff["poss_last10"]
    team_eff["drtg_last10"] = 100.0 * team_eff["pa_last10"] / team_eff["poss_last10"]
    team_eff["netrtg_last10"] = team_eff["ortg_last10"] - team_eff["drtg_last10"]

    team_eff.loc[
        team_eff["poss_last10"] <= 0, ["ortg_last10", "drtg_last10", "netrtg_last10"]
    ] = np.nan

    home_eff = team_eff[["game_id", "team_id", "netrtg_last10"]].rename(
        columns={"team_id": "team_id_home", "netrtg_last10": "home_netrtg_last10"}
    )
    away_eff = team_eff[["game_id", "team_id", "netrtg_last10"]].rename(
        columns={"team_id": "team_id_away", "netrtg_last10": "away_netrtg_last10"}
    )

    df = df.merge(home_eff, on=["game_id", "team_id_home"], how="left")
    df = df.merge(away_eff, on=["game_id", "team_id_away"], how="left")
    df["netrtg_diff_last10"] = df["home_netrtg_last10"] - df["away_netrtg_last10"]
    return df


def add_four_factor_features(
    conn: duckdb.DuckDBPyConnection, df: pd.DataFrame, base_table: str
) -> pd.DataFrame:
    print("Computing rolling Four Factors features...")
    required_cols = {
        "game_id",
        "game_date",
        "team_id_home",
        "team_id_away",
        "fgm_home",
        "fg3m_home",
        "fga_home",
        "fta_home",
        "oreb_home",
        "tov_home",
        "dreb_home",
        "fgm_away",
        "fg3m_away",
        "fga_away",
        "fta_away",
        "oreb_away",
        "tov_away",
        "dreb_away",
    }

    available_cols = set(get_columns(conn, base_table))
    if not required_cols.issubset(available_cols):
        print(
            "[WARN] Skipping Four Factors features due to missing columns. "
            f"Missing: {sorted(required_cols - available_cols)}"
        )
        for col in [
            "home_efg_last10",
            "home_tov_pct_last10",
            "home_orb_pct_last10",
            "home_ftr_last10",
            "away_efg_last10",
            "away_tov_pct_last10",
            "away_orb_pct_last10",
            "away_ftr_last10",
            "efg_diff_last10",
            "tov_pct_diff_last10",
            "orb_pct_diff_last10",
            "ftr_diff_last10",
        ]:
            df[col] = np.nan
        return df

    ff_games = safe_query(
        conn,
        f"""
        SELECT
            CAST(game_id AS VARCHAR) AS game_id,
            CAST(game_date AS DATE) AS game_date,
            CAST(team_id_home AS VARCHAR) AS team_id_home,
            CAST(team_id_away AS VARCHAR) AS team_id_away,
            TRY_CAST(fgm_home AS DOUBLE) AS fgm_home,
            TRY_CAST(fg3m_home AS DOUBLE) AS fg3m_home,
            TRY_CAST(fga_home AS DOUBLE) AS fga_home,
            TRY_CAST(fta_home AS DOUBLE) AS fta_home,
            TRY_CAST(oreb_home AS DOUBLE) AS oreb_home,
            TRY_CAST(tov_home AS DOUBLE) AS tov_home,
            TRY_CAST(dreb_home AS DOUBLE) AS dreb_home,
            TRY_CAST(fgm_away AS DOUBLE) AS fgm_away,
            TRY_CAST(fg3m_away AS DOUBLE) AS fg3m_away,
            TRY_CAST(fga_away AS DOUBLE) AS fga_away,
            TRY_CAST(fta_away AS DOUBLE) AS fta_away,
            TRY_CAST(oreb_away AS DOUBLE) AS oreb_away,
            TRY_CAST(tov_away AS DOUBLE) AS tov_away,
            TRY_CAST(dreb_away AS DOUBLE) AS dreb_away
        FROM {base_table}
        ORDER BY game_date, game_id
        """,
        label="load four-factor source data",
        tables_to_debug=[base_table],
    )

    home_ff_long = ff_games[
        [
            "game_id",
            "game_date",
            "team_id_home",
            "fgm_home",
            "fg3m_home",
            "fga_home",
            "fta_home",
            "oreb_home",
            "tov_home",
            "dreb_away",
        ]
    ].rename(
        columns={
            "team_id_home": "team_id",
            "fgm_home": "fgm",
            "fg3m_home": "fg3m",
            "fga_home": "fga",
            "fta_home": "fta",
            "oreb_home": "oreb",
            "tov_home": "tov",
            "dreb_away": "opp_dreb",
        }
    )
    away_ff_long = ff_games[
        [
            "game_id",
            "game_date",
            "team_id_away",
            "fgm_away",
            "fg3m_away",
            "fga_away",
            "fta_away",
            "oreb_away",
            "tov_away",
            "dreb_home",
        ]
    ].rename(
        columns={
            "team_id_away": "team_id",
            "fgm_away": "fgm",
            "fg3m_away": "fg3m",
            "fga_away": "fga",
            "fta_away": "fta",
            "oreb_away": "oreb",
            "tov_away": "tov",
            "dreb_home": "opp_dreb",
        }
    )

    ff_long = pd.concat([home_ff_long, away_ff_long], ignore_index=True)
    ff_long = ff_long.sort_values(["team_id", "game_date", "game_id"])
    grouped = ff_long.groupby("team_id")

    for col in ["fgm", "fg3m", "fga", "fta", "oreb", "tov", "opp_dreb"]:
        ff_long[f"{col}_last10"] = grouped[col].transform(
            lambda s: s.shift(1).rolling(window=10, min_periods=5).sum()
        )

    ff_long["efg_last10"] = (
        ff_long["fgm_last10"] + 0.5 * ff_long["fg3m_last10"]
    ) / ff_long["fga_last10"]
    ff_long["tov_pct_last10"] = ff_long["tov_last10"] / (
        ff_long["fga_last10"] + 0.44 * ff_long["fta_last10"] + ff_long["tov_last10"]
    )
    ff_long["orb_pct_last10"] = ff_long["oreb_last10"] / (
        ff_long["oreb_last10"] + ff_long["opp_dreb_last10"]
    )
    ff_long["ftr_last10"] = ff_long["fta_last10"] / ff_long["fga_last10"]

    ff_long.loc[ff_long["fga_last10"] <= 0, ["efg_last10", "ftr_last10"]] = np.nan
    ff_long.loc[
        (ff_long["fga_last10"] + 0.44 * ff_long["fta_last10"] + ff_long["tov_last10"])
        <= 0,
        "tov_pct_last10",
    ] = np.nan
    ff_long.loc[
        (ff_long["oreb_last10"] + ff_long["opp_dreb_last10"]) <= 0, "orb_pct_last10"
    ] = np.nan

    home_ff = ff_long[
        [
            "game_id",
            "team_id",
            "efg_last10",
            "tov_pct_last10",
            "orb_pct_last10",
            "ftr_last10",
        ]
    ].rename(
        columns={
            "team_id": "team_id_home",
            "efg_last10": "home_efg_last10",
            "tov_pct_last10": "home_tov_pct_last10",
            "orb_pct_last10": "home_orb_pct_last10",
            "ftr_last10": "home_ftr_last10",
        }
    )
    away_ff = ff_long[
        [
            "game_id",
            "team_id",
            "efg_last10",
            "tov_pct_last10",
            "orb_pct_last10",
            "ftr_last10",
        ]
    ].rename(
        columns={
            "team_id": "team_id_away",
            "efg_last10": "away_efg_last10",
            "tov_pct_last10": "away_tov_pct_last10",
            "orb_pct_last10": "away_orb_pct_last10",
            "ftr_last10": "away_ftr_last10",
        }
    )

    df = df.merge(home_ff, on=["game_id", "team_id_home"], how="left")
    df = df.merge(away_ff, on=["game_id", "team_id_away"], how="left")

    df["efg_diff_last10"] = df["home_efg_last10"] - df["away_efg_last10"]
    df["tov_pct_diff_last10"] = df["home_tov_pct_last10"] - df["away_tov_pct_last10"]
    df["orb_pct_diff_last10"] = df["home_orb_pct_last10"] - df["away_orb_pct_last10"]
    df["ftr_diff_last10"] = df["home_ftr_last10"] - df["away_ftr_last10"]
    return df


def ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    print("Ensuring required training features exist...")
    for col in IDENTITY_COLUMNS + REQUIRED_MODEL_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["season_id"] = pd.to_numeric(df["season_id"], errors="coerce")
    df["home_win"] = pd.to_numeric(df["home_win"], errors="coerce")
    df["home_win"] = np.where(
        df["home_win"] >= 0.5, 1, np.where(df["home_win"] < 0.5, 0, np.nan)
    )

    df["home_net_last5"] = pd.to_numeric(
        df["home_avg_pts_for_last5"], errors="coerce"
    ) - pd.to_numeric(df["home_avg_pts_against_last5"], errors="coerce")
    df["away_net_last5"] = pd.to_numeric(
        df["away_avg_pts_for_last5"], errors="coerce"
    ) - pd.to_numeric(df["away_avg_pts_against_last5"], errors="coerce")
    df["net_diff_last5"] = df["home_net_last5"] - df["away_net_last5"]
    df["win_pct_diff_last5"] = pd.to_numeric(
        df["home_win_pct_last5"], errors="coerce"
    ) - pd.to_numeric(df["away_win_pct_last5"], errors="coerce")

    df = df[df["game_date"].notna()].copy()
    df = df[df["home_win"].isin([0, 1])].copy()
    df = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    ordered_cols = IDENTITY_COLUMNS + REQUIRED_MODEL_FEATURES
    for extra in df.columns:
        if extra not in ordered_cols:
            ordered_cols.append(extra)
    return df[ordered_cols]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    db_path = repo_root / "data" / "nba.duckdb"
    if not db_path.is_file():
        raise FileNotFoundError(f"DuckDB file not found: {db_path}")

    conn = duckdb.connect(str(db_path))
    try:
        print(f"Connected to DuckDB: {db_path}")
        print("Building schema-safe base table...")
        base_table = build_model_base_enriched(conn)

        base_games = safe_query(
            conn,
            f"""
            SELECT
                CAST(game_id AS VARCHAR) AS game_id,
                CAST(game_date AS DATE) AS game_date,
                COALESCE(TRY_CAST(season_id AS INTEGER), EXTRACT(YEAR FROM CAST(game_date AS DATE))) AS season_id,
                CAST(team_id_home AS VARCHAR) AS team_id_home,
                CAST(team_id_away AS VARCHAR) AS team_id_away,
                CASE
                    WHEN TRY_CAST(home_win AS DOUBLE) >= 0.5 THEN 1
                    WHEN TRY_CAST(home_win AS DOUBLE) < 0.5 THEN 0
                    ELSE NULL
                END AS home_win,
                TRY_CAST(pts_home AS DOUBLE) AS pts_home,
                TRY_CAST(pts_away AS DOUBLE) AS pts_away
            FROM {base_table}
            ORDER BY game_date, game_id
            """,
            label="load base games for Elo",
            tables_to_debug=[base_table],
        )

        df = load_or_build_base_rolling_features(conn, base_table)
        df = add_elo_features(df, base_games)
        df = add_rest_day_features(conn, df, base_table)
        df = add_pace_features(conn, df, base_table)
        df = add_four_factor_features(conn, df, base_table)
        df = ensure_required_columns(df)

        if df.empty:
            raise RuntimeError(
                "Feature pipeline produced zero rows after validation. "
                "Check source data coverage and target derivation."
            )

        print(
            f"Saving final_features table with {len(df)} rows and {len(df.columns)} columns..."
        )
        conn.register("features_temp", df)
        conn.execute("""
            CREATE OR REPLACE TABLE final_features AS
            SELECT *
            FROM features_temp
            """)
        print("Feature pipeline complete.")

    except Exception as exc:
        print(f"[ERROR] build_features pipeline failed: {exc}")
        print_available_columns(
            conn, ["model_base", "game", "game_features_clean", "team_game_long"]
        )
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
