from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

INPUT_CSV = Path("data/processed_games.csv")
OUTPUT_CSV = Path("data/final_features.csv")

INITIAL_ELO = 1500.0
K = 20.0
HOME_ADVANTAGE = 100.0
CARRYOVER = 0.75

REQUIRED_FEATURES = [
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


def print_columns(df: pd.DataFrame, label: str) -> None:
    print(f"[DEBUG] available columns in {label}: {list(df.columns)}")


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def parse_binary(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.where(numeric >= 0.5, 1, np.where(numeric < 0.5, 0, np.nan)), index=series.index)

    missing = out.isna()
    if missing.any():
        text = series.astype(str).str.strip().str.upper()
        out.loc[missing & text.isin(["W", "WIN", "TRUE", "T"])] = 1
        out.loc[missing & text.isin(["L", "LOSS", "FALSE", "F"])] = 0
    return out


def ensure_column(df: pd.DataFrame, col: str, fill_value=np.nan) -> None:
    if col not in df.columns:
        df[col] = fill_value


def normalize_base(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    game_id_col = first_existing(df, ["game_id", "id"])
    game_date_col = first_existing(df, ["game_date", "game_date_est", "date_game", "date"])
    season_col = first_existing(df, ["season_id", "season", "season_year"])
    home_id_col = first_existing(df, ["team_id_home", "home_team_id", "team_home_id"])
    away_id_col = first_existing(df, ["team_id_away", "away_team_id", "team_away_id"])
    home_name_col = first_existing(df, ["team_name_home", "home_team_name", "home_team"])
    away_name_col = first_existing(df, ["team_name_away", "away_team_name", "away_team"])

    if game_date_col is None:
        print_columns(df, "processed_games.csv")
        raise RuntimeError("Missing game date column.")

    if game_id_col is None:
        df["game_id"] = np.arange(1, len(df) + 1).astype(str)
    else:
        df["game_id"] = df[game_id_col].astype(str)

    df["game_date"] = pd.to_datetime(df[game_date_col], errors="coerce")
    if season_col is not None:
        df["season_id"] = pd.to_numeric(df[season_col], errors="coerce")
    else:
        df["season_id"] = df["game_date"].dt.year

    if home_id_col is not None:
        df["team_id_home"] = df[home_id_col].astype(str)
    elif home_name_col is not None:
        df["team_id_home"] = df[home_name_col].astype(str)
    else:
        print_columns(df, "processed_games.csv")
        raise RuntimeError("Missing home team identifier.")

    if away_id_col is not None:
        df["team_id_away"] = df[away_id_col].astype(str)
    elif away_name_col is not None:
        df["team_id_away"] = df[away_name_col].astype(str)
    else:
        print_columns(df, "processed_games.csv")
        raise RuntimeError("Missing away team identifier.")

    if "home_win" in df.columns:
        df["home_win"] = parse_binary(df["home_win"])
    elif "wl_home" in df.columns:
        df["home_win"] = parse_binary(df["wl_home"])
    elif "away_win" in df.columns:
        df["home_win"] = 1 - parse_binary(df["away_win"])
    elif "wl_away" in df.columns:
        df["home_win"] = 1 - parse_binary(df["wl_away"])
    else:
        print_columns(df, "processed_games.csv")
        raise RuntimeError("Missing target columns for home_win derivation.")

    df["away_win"] = 1 - df["home_win"]

    for c in [
        "pts_home",
        "pts_away",
        "fgm_home",
        "fg3m_home",
        "fga_home",
        "fta_home",
        "oreb_home",
        "dreb_home",
        "tov_home",
        "fgm_away",
        "fg3m_away",
        "fga_away",
        "fta_away",
        "oreb_away",
        "dreb_away",
        "tov_away",
    ]:
        ensure_column(df, c, np.nan)
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    return df


def build_team_long(df: pd.DataFrame) -> pd.DataFrame:
    home = df[
        [
            "game_id",
            "game_date",
            "season_id",
            "team_id_home",
            "team_id_away",
            "pts_home",
            "pts_away",
            "home_win",
        ]
    ].rename(
        columns={
            "team_id_home": "team_id",
            "team_id_away": "opponent_id",
            "pts_home": "points_for",
            "pts_away": "points_against",
            "home_win": "win",
        }
    )
    home["is_home"] = 1

    away = df[
        [
            "game_id",
            "game_date",
            "season_id",
            "team_id_away",
            "team_id_home",
            "pts_away",
            "pts_home",
            "away_win",
        ]
    ].rename(
        columns={
            "team_id_away": "team_id",
            "team_id_home": "opponent_id",
            "pts_away": "points_for",
            "pts_home": "points_against",
            "away_win": "win",
        }
    )
    away["is_home"] = 0

    team_long = pd.concat([home, away], ignore_index=True)
    team_long = team_long.sort_values(["team_id", "game_date", "game_id"]).reset_index(drop=True)
    return team_long


def add_rolling_form_features(df: pd.DataFrame, team_long: pd.DataFrame) -> pd.DataFrame:
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

    home_roll = team_long[["game_id", "team_id", "avg_pts_for_last5", "avg_pts_against_last5", "win_pct_last5"]].rename(
        columns={
            "team_id": "team_id_home",
            "avg_pts_for_last5": "home_avg_pts_for_last5",
            "avg_pts_against_last5": "home_avg_pts_against_last5",
            "win_pct_last5": "home_win_pct_last5",
        }
    )
    away_roll = team_long[["game_id", "team_id", "avg_pts_for_last5", "avg_pts_against_last5", "win_pct_last5"]].rename(
        columns={
            "team_id": "team_id_away",
            "avg_pts_for_last5": "away_avg_pts_for_last5",
            "avg_pts_against_last5": "away_avg_pts_against_last5",
            "win_pct_last5": "away_win_pct_last5",
        }
    )

    out = df.merge(home_roll, on=["game_id", "team_id_home"], how="left")
    out = out.merge(away_roll, on=["game_id", "team_id_away"], how="left")
    return out


def add_elo_features(df: pd.DataFrame) -> pd.DataFrame:
    elo_ratings: dict[str, float] = {}
    home_elo_pre: list[float] = []
    away_elo_pre: list[float] = []
    current_season = None

    for _, row in df.iterrows():
        season = row["season_id"]
        if current_season is not None and pd.notna(season) and season != current_season:
            for team in elo_ratings:
                elo_ratings[team] = CARRYOVER * elo_ratings[team] + (1 - CARRYOVER) * INITIAL_ELO

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

        expected_home = 1 / (1 + 10 ** ((r_away - (r_home + HOME_ADVANTAGE)) / 400))
        actual_home = row["home_win"]
        if pd.isna(actual_home):
            actual_home = 0.5

        pts_home = row["pts_home"] if pd.notna(row["pts_home"]) else 0.0
        pts_away = row["pts_away"] if pd.notna(row["pts_away"]) else 0.0
        point_diff = abs(float(pts_home) - float(pts_away))
        mov_multiplier = np.log(point_diff + 1) * (2.2 / ((r_home - r_away) * 0.001 + 2.2))

        elo_ratings[home] = r_home + K * mov_multiplier * (float(actual_home) - expected_home)
        elo_ratings[away] = r_away + K * mov_multiplier * ((1 - float(actual_home)) - (1 - expected_home))

    out = df.copy()
    out["home_elo_pre"] = home_elo_pre
    out["away_elo_pre"] = away_elo_pre
    out["elo_diff"] = out["home_elo_pre"] - out["away_elo_pre"]
    return out


def add_rest_features(df: pd.DataFrame, team_long: pd.DataFrame) -> pd.DataFrame:
    rest = team_long[["game_id", "game_date", "team_id"]].copy()
    rest["prev_game_date"] = rest.groupby("team_id")["game_date"].shift(1)
    rest["rest_days"] = (rest["game_date"] - rest["prev_game_date"]).dt.days

    home_rest = rest[["game_id", "team_id", "rest_days"]].rename(
        columns={"team_id": "team_id_home", "rest_days": "home_rest_days"}
    )
    away_rest = rest[["game_id", "team_id", "rest_days"]].rename(
        columns={"team_id": "team_id_away", "rest_days": "away_rest_days"}
    )

    out = df.merge(home_rest, on=["game_id", "team_id_home"], how="left")
    out = out.merge(away_rest, on=["game_id", "team_id_away"], how="left")
    out["rest_diff"] = out["home_rest_days"] - out["away_rest_days"]
    out["home_b2b"] = (pd.to_numeric(out["home_rest_days"], errors="coerce") <= 1).astype(float)
    out["away_b2b"] = (pd.to_numeric(out["away_rest_days"], errors="coerce") <= 1).astype(float)
    out["b2b_diff"] = out["home_b2b"] - out["away_b2b"]
    return out


def add_pace_features(df: pd.DataFrame) -> pd.DataFrame:
    needed = [
        "fga_home",
        "fta_home",
        "oreb_home",
        "tov_home",
        "fga_away",
        "fta_away",
        "oreb_away",
        "tov_away",
        "pts_home",
        "pts_away",
    ]
    if any(col not in df.columns for col in needed):
        out = df.copy()
        out["home_netrtg_last10"] = np.nan
        out["away_netrtg_last10"] = np.nan
        out["netrtg_diff_last10"] = np.nan
        return out

    pace = df[
        [
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
        ]
    ].copy()

    pace["possessions"] = 0.5 * (
        (pace["fga_home"] + 0.44 * pace["fta_home"] - pace["oreb_home"] + pace["tov_home"])
        + (pace["fga_away"] + 0.44 * pace["fta_away"] - pace["oreb_away"] + pace["tov_away"])
    )

    home_long = pace[["game_id", "game_date", "team_id_home", "pts_home", "pts_away", "possessions"]].rename(
        columns={"team_id_home": "team_id", "pts_home": "points_for", "pts_away": "points_against"}
    )
    away_long = pace[["game_id", "game_date", "team_id_away", "pts_away", "pts_home", "possessions"]].rename(
        columns={"team_id_away": "team_id", "pts_away": "points_for", "pts_home": "points_against"}
    )
    team_eff = pd.concat([home_long, away_long], ignore_index=True)
    team_eff = team_eff.sort_values(["team_id", "game_date", "game_id"])
    grouped = team_eff.groupby("team_id")

    team_eff["pf_last10"] = grouped["points_for"].transform(lambda s: s.shift(1).rolling(window=10, min_periods=5).sum())
    team_eff["pa_last10"] = grouped["points_against"].transform(lambda s: s.shift(1).rolling(window=10, min_periods=5).sum())
    team_eff["poss_last10"] = grouped["possessions"].transform(lambda s: s.shift(1).rolling(window=10, min_periods=5).sum())

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

    out = df.merge(home_eff, on=["game_id", "team_id_home"], how="left")
    out = out.merge(away_eff, on=["game_id", "team_id_away"], how="left")
    out["netrtg_diff_last10"] = out["home_netrtg_last10"] - out["away_netrtg_last10"]
    return out


def add_four_factor_features(df: pd.DataFrame) -> pd.DataFrame:
    needed = [
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
    ]
    if any(col not in df.columns for col in needed):
        out = df.copy()
        for c in [
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
            out[c] = np.nan
        return out

    ff = df[
        [
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
        ]
    ].copy()

    home_long = ff[
        ["game_id", "game_date", "team_id_home", "fgm_home", "fg3m_home", "fga_home", "fta_home", "oreb_home", "tov_home", "dreb_away"]
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
    away_long = ff[
        ["game_id", "game_date", "team_id_away", "fgm_away", "fg3m_away", "fga_away", "fta_away", "oreb_away", "tov_away", "dreb_home"]
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
    ff_long = pd.concat([home_long, away_long], ignore_index=True)
    ff_long = ff_long.sort_values(["team_id", "game_date", "game_id"])
    grouped = ff_long.groupby("team_id")

    for col in ["fgm", "fg3m", "fga", "fta", "oreb", "tov", "opp_dreb"]:
        ff_long[f"{col}_last10"] = grouped[col].transform(lambda s: s.shift(1).rolling(window=10, min_periods=5).sum())

    ff_long["efg_last10"] = (ff_long["fgm_last10"] + 0.5 * ff_long["fg3m_last10"]) / ff_long["fga_last10"]
    ff_long["tov_pct_last10"] = ff_long["tov_last10"] / (ff_long["fga_last10"] + 0.44 * ff_long["fta_last10"] + ff_long["tov_last10"])
    ff_long["orb_pct_last10"] = ff_long["oreb_last10"] / (ff_long["oreb_last10"] + ff_long["opp_dreb_last10"])
    ff_long["ftr_last10"] = ff_long["fta_last10"] / ff_long["fga_last10"]

    ff_long.loc[ff_long["fga_last10"] <= 0, ["efg_last10", "ftr_last10"]] = np.nan
    ff_long.loc[(ff_long["fga_last10"] + 0.44 * ff_long["fta_last10"] + ff_long["tov_last10"]) <= 0, "tov_pct_last10"] = np.nan
    ff_long.loc[(ff_long["oreb_last10"] + ff_long["opp_dreb_last10"]) <= 0, "orb_pct_last10"] = np.nan

    home_ff = ff_long[["game_id", "team_id", "efg_last10", "tov_pct_last10", "orb_pct_last10", "ftr_last10"]].rename(
        columns={
            "team_id": "team_id_home",
            "efg_last10": "home_efg_last10",
            "tov_pct_last10": "home_tov_pct_last10",
            "orb_pct_last10": "home_orb_pct_last10",
            "ftr_last10": "home_ftr_last10",
        }
    )
    away_ff = ff_long[["game_id", "team_id", "efg_last10", "tov_pct_last10", "orb_pct_last10", "ftr_last10"]].rename(
        columns={
            "team_id": "team_id_away",
            "efg_last10": "away_efg_last10",
            "tov_pct_last10": "away_tov_pct_last10",
            "orb_pct_last10": "away_orb_pct_last10",
            "ftr_last10": "away_ftr_last10",
        }
    )

    out = df.merge(home_ff, on=["game_id", "team_id_home"], how="left")
    out = out.merge(away_ff, on=["game_id", "team_id_away"], how="left")
    out["efg_diff_last10"] = out["home_efg_last10"] - out["away_efg_last10"]
    out["tov_pct_diff_last10"] = out["home_tov_pct_last10"] - out["away_tov_pct_last10"]
    out["orb_pct_diff_last10"] = out["home_orb_pct_last10"] - out["away_orb_pct_last10"]
    out["ftr_diff_last10"] = out["home_ftr_last10"] - out["away_ftr_last10"]
    return out


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["home_net_last5"] = pd.to_numeric(out["home_avg_pts_for_last5"], errors="coerce") - pd.to_numeric(
        out["home_avg_pts_against_last5"], errors="coerce"
    )
    out["away_net_last5"] = pd.to_numeric(out["away_avg_pts_for_last5"], errors="coerce") - pd.to_numeric(
        out["away_avg_pts_against_last5"], errors="coerce"
    )
    out["net_diff_last5"] = out["home_net_last5"] - out["away_net_last5"]
    out["win_pct_diff_last5"] = pd.to_numeric(out["home_win_pct_last5"], errors="coerce") - pd.to_numeric(
        out["away_win_pct_last5"], errors="coerce"
    )

    for c in ["game_id", "game_date", "season_id", "team_id_home", "team_id_away", "home_win"] + REQUIRED_FEATURES:
        ensure_column(out, c, np.nan)

    out["home_win"] = parse_binary(out["home_win"])
    out = out[out["game_date"].notna()].copy()
    out = out[out["home_win"].isin([0, 1])].copy()
    out = out.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    final_cols = ["game_id", "game_date", "season_id", "team_id_home", "team_id_away", "home_win"] + REQUIRED_FEATURES
    return out[final_cols]


def main() -> None:
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f"Missing input file: {INPUT_CSV}")

    print(f"Loading source data: {INPUT_CSV}")
    raw = pd.read_csv(INPUT_CSV)
    base = normalize_base(raw)
    team_long = build_team_long(base)

    features = add_rolling_form_features(base, team_long)
    features = add_elo_features(features)
    features = add_rest_features(features, team_long)
    features = add_pace_features(features)
    features = add_four_factor_features(features)
    final_features = finalize(features)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    final_features.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved final features: {OUTPUT_CSV} ({len(final_features)} rows, {len(final_features.columns)} columns)")


if __name__ == "__main__":
    main()

