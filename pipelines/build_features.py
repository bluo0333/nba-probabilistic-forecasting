from pathlib import Path
import duckdb
import pandas as pd
import numpy as np


def main():
    repo_root = Path(__file__).resolve().parents[1]
    db_path = repo_root / "data" / "nba.duckdb"

    conn = duckdb.connect(str(db_path))

    print("Loading base game data...")
    games = conn.execute("""
        SELECT
            game_id,
            game_date,
            team_id_home,
            team_id_away,
            home_win
        FROM model_base
        ORDER BY game_date
    """).df()

    #=============================================
    # Build MOV-Adjusted Elo

    INITIAL_ELO = 1500
    K = 20
    HOME_ADVANTAGE = 100

    elo_ratings = {}
    home_elo_pre = []
    away_elo_pre = []

    print("Computing MOV-adjusted Elo ratings...")

    games_full = conn.execute("""
        SELECT
            game_id,
            game_date,
            season_id,
            team_id_home,
            team_id_away,
            home_win,
            pts_home,
            pts_away
        FROM model_base
        ORDER BY game_date
    """).df()

    print("Computing MOV-adjusted Elo ratings with offseason regression...")

    # offseason strength retention

    CARRYOVER = 0.75

    current_season = None

    for _, row in games_full.iterrows():

        season = row["season_id"]

        # Detect season change
        if current_season is not None and season != current_season:
            for team in elo_ratings:
                elo_ratings[team] = (
                    CARRYOVER * elo_ratings[team] +
                    (1 - CARRYOVER) * INITIAL_ELO
                )

        current_season = season

        home = row["team_id_home"]
        away = row["team_id_away"]

        if home not in elo_ratings:
            elo_ratings[home] = INITIAL_ELO
        if away not in elo_ratings:
            elo_ratings[away] = INITIAL_ELO

        R_home = elo_ratings[home]
        R_away = elo_ratings[away]

        home_elo_pre.append(R_home)
        away_elo_pre.append(R_away)

        expected_home = 1 / (1 + 10 ** ((R_away - (R_home + HOME_ADVANTAGE)) / 400))
        actual_home = row["home_win"]

        point_diff = abs(row["pts_home"] - row["pts_away"])

        mov_multiplier = (
            np.log(point_diff + 1) *
            (2.2 / ((R_home - R_away) * 0.001 + 2.2))
        )

        elo_ratings[home] = R_home + K * mov_multiplier * (actual_home - expected_home)
        elo_ratings[away] = R_away + K * mov_multiplier * ((1 - actual_home) - (1 - expected_home))

    games_full["home_elo_pre"] = home_elo_pre
    games_full["away_elo_pre"] = away_elo_pre
    games_full["elo_diff"] = games_full["home_elo_pre"] - games_full["away_elo_pre"]


    #=============================================
    # Load Rolling Feature Data

    print("Loading rolling feature data...")
    df = conn.execute("SELECT * FROM game_features_clean").df()

    # Merge MOV-adjusted Elo
    print("Merging MOV-adjusted Elo...")
    df = df.merge(
        games_full[["game_id", "home_elo_pre", "away_elo_pre", "elo_diff"]],
        on="game_id",
        how="inner"
    )


    #=============================================
    # Rest Day Features

    print("Computing rest day features...")

    team_games = conn.execute("""
        SELECT
            game_id,
            game_date,
            team_id,
            is_home
        FROM team_game_long
        ORDER BY team_id, game_date
    """).df()

    team_games["prev_game_date"] = (
        team_games.groupby("team_id")["game_date"].shift(1)
    )

    team_games["rest_days"] = (
        team_games["game_date"] - team_games["prev_game_date"]
    ).dt.days

    home_rest = team_games[["game_id", "team_id", "rest_days"]].rename(
        columns={
            "team_id": "team_id_home",
            "rest_days": "home_rest_days"
        }
    )

    away_rest = team_games[["game_id", "team_id", "rest_days"]].rename(
        columns={
            "team_id": "team_id_away",
            "rest_days": "away_rest_days"
        }
    )

    df = df.merge(home_rest, on=["game_id", "team_id_home"], how="left")
    df = df.merge(away_rest, on=["game_id", "team_id_away"], how="left")

    df["rest_diff"] = df["home_rest_days"] - df["away_rest_days"]

    df["home_b2b"] = (df["home_rest_days"] <= 1).astype(int)
    df["away_b2b"] = (df["away_rest_days"] <= 1).astype(int)
    df["b2b_diff"] = df["home_b2b"] - df["away_b2b"]

    #=============================================
    # Pace-Adjusted Efficiency Features

    print("Computing pace-adjusted efficiency features...")

    pace_games = conn.execute("""
        SELECT
            game_id,
            game_date,
            team_id_home,
            team_id_away,
            pts_home,
            pts_away,
            fga_home,
            fta_home,
            oreb_home,
            tov_home,
            fga_away,
            fta_away,
            oreb_away,
            tov_away
        FROM game
        ORDER BY game_date
    """).df()

    pace_games["possessions"] = 0.5 * (
        (pace_games["fga_home"] + 0.44 * pace_games["fta_home"] - pace_games["oreb_home"] + pace_games["tov_home"]) +
        (pace_games["fga_away"] + 0.44 * pace_games["fta_away"] - pace_games["oreb_away"] + pace_games["tov_away"])
    )

    home_eff_long = pace_games[["game_id", "game_date", "team_id_home", "pts_home", "pts_away", "possessions"]].rename(
        columns={
            "team_id_home": "team_id",
            "pts_home": "points_for",
            "pts_away": "points_against"
        }
    )

    away_eff_long = pace_games[["game_id", "game_date", "team_id_away", "pts_away", "pts_home", "possessions"]].rename(
        columns={
            "team_id_away": "team_id",
            "pts_away": "points_for",
            "pts_home": "points_against"
        }
    )

    team_eff = pd.concat([home_eff_long, away_eff_long], ignore_index=True)
    team_eff = team_eff.sort_values(["team_id", "game_date", "game_id"])

    grouped_eff = team_eff.groupby("team_id")

    team_eff["pf_last10"] = grouped_eff["points_for"].transform(
        lambda s: s.shift(1).rolling(window=10, min_periods=5).sum()
    )
    team_eff["pa_last10"] = grouped_eff["points_against"].transform(
        lambda s: s.shift(1).rolling(window=10, min_periods=5).sum()
    )
    team_eff["poss_last10"] = grouped_eff["possessions"].transform(
        lambda s: s.shift(1).rolling(window=10, min_periods=5).sum()
    )

    team_eff["ortg_last10"] = 100 * team_eff["pf_last10"] / team_eff["poss_last10"]
    team_eff["drtg_last10"] = 100 * team_eff["pa_last10"] / team_eff["poss_last10"]
    team_eff["netrtg_last10"] = team_eff["ortg_last10"] - team_eff["drtg_last10"]

    team_eff.loc[team_eff["poss_last10"] <= 0, ["ortg_last10", "drtg_last10", "netrtg_last10"]] = np.nan

    home_eff = team_eff[["game_id", "team_id", "netrtg_last10"]].rename(
        columns={"team_id": "team_id_home", "netrtg_last10": "home_netrtg_last10"}
    )

    away_eff = team_eff[["game_id", "team_id", "netrtg_last10"]].rename(
        columns={"team_id": "team_id_away", "netrtg_last10": "away_netrtg_last10"}
    )

    df = df.merge(home_eff, on=["game_id", "team_id_home"], how="left")
    df = df.merge(away_eff, on=["game_id", "team_id_away"], how="left")

    df["netrtg_diff_last10"] = df["home_netrtg_last10"] - df["away_netrtg_last10"]

    #=============================================
    # Rolling Four Factors Features

    print("Computing rolling Four Factors features...")

    four_factors_games = conn.execute("""
        SELECT
            game_id,
            game_date,
            team_id_home,
            team_id_away,
            fgm_home,
            fg3m_home,
            fga_home,
            fta_home,
            oreb_home,
            tov_home,
            dreb_home,
            fgm_away,
            fg3m_away,
            fga_away,
            fta_away,
            oreb_away,
            tov_away,
            dreb_away
        FROM game
        ORDER BY game_date
    """).df()

    home_ff_long = four_factors_games[
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

    away_ff_long = four_factors_games[
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

    grouped_ff = ff_long.groupby("team_id")
    rolling_sources = ["fgm", "fg3m", "fga", "fta", "oreb", "tov", "opp_dreb"]

    for col in rolling_sources:
        ff_long[f"{col}_last10"] = grouped_ff[col].transform(
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
        (ff_long["fga_last10"] + 0.44 * ff_long["fta_last10"] + ff_long["tov_last10"]) <= 0,
        "tov_pct_last10",
    ] = np.nan
    ff_long.loc[(ff_long["oreb_last10"] + ff_long["opp_dreb_last10"]) <= 0, "orb_pct_last10"] = np.nan

    home_ff = ff_long[
        ["game_id", "team_id", "efg_last10", "tov_pct_last10", "orb_pct_last10", "ftr_last10"]
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
        ["game_id", "team_id", "efg_last10", "tov_pct_last10", "orb_pct_last10", "ftr_last10"]
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

    #=============================================
    # Differential Features

    df["home_net_last5"] = (
        df["home_avg_pts_for_last5"] -
        df["home_avg_pts_against_last5"]
    )

    df["away_net_last5"] = (
        df["away_avg_pts_for_last5"] -
        df["away_avg_pts_against_last5"]
    )

    df["net_diff_last5"] = df["home_net_last5"] - df["away_net_last5"]

    df["win_pct_diff_last5"] = (
        df["home_win_pct_last5"] -
        df["away_win_pct_last5"]
    )

    print("Dropping NaNs...")
    df = df.dropna()

    print("Saving final_features table...")
    conn.register("features_temp", df)

    conn.execute("""
        CREATE OR REPLACE TABLE final_features AS
        SELECT *
        FROM features_temp
    """)

    print("Feature pipeline complete.")


if __name__ == "__main__":
    main()
