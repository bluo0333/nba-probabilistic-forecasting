from __future__ import annotations

import math
from typing import Any

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
STATE_DEFAULTS = {
    "elo_pre": 1500.0,
    "avg_pts_for_last5": 110.0,
    "avg_pts_against_last5": 110.0,
    "win_pct_last5": 0.5,
    "netrtg_last10": 0.0,
    "efg_last10": 0.53,
    "tov_pct_last10": 0.13,
    "orb_pct_last10": 0.28,
    "ftr_last10": 0.20,
}


def _safe_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if pd.isna(out):
        return float(default)
    return out


def _state_value(state: dict[str, Any], key: str) -> float:
    return _safe_float(state.get(key), STATE_DEFAULTS[key])


def normalize_name(value: str) -> str:
    return str(value or "").strip().lower()


def american_to_implied_probability(odds: float) -> float:
    if odds == 0:
        raise ValueError("Odds cannot be zero.")
    if odds < 0:
        return -odds / (-odds + 100.0)
    return 100.0 / (odds + 100.0)


def normal_cdf(value: float, mean: float, std_dev: float) -> float:
    if std_dev <= 0:
        return 1.0 if value >= mean else 0.0
    z = (value - mean) / (std_dev * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def pick_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def latest_feature_row(features_df: pd.DataFrame, player_norm: str) -> pd.Series | None:
    rows = features_df[features_df["player_name_norm"] == player_norm].copy()
    if rows.empty:
        return None
    order_cols = [column for column in ["date", "game_id"] if column in rows.columns]
    if order_cols:
        rows = rows.sort_values(order_cols, ascending=True)
    return rows.iloc[-1]


def predict_mean_from_model(model: Any, row: pd.Series) -> float | None:
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        feature_names = [
            "rolling_points_10",
            "rolling_assists_10",
            "rolling_rebounds_10",
            "rolling_minutes_10",
        ]

    values: dict[str, float] = {}
    for feature_name in feature_names:
        if feature_name not in row.index:
            return None
        value = row.get(feature_name)
        if value is None or pd.isna(value):
            return None
        values[str(feature_name)] = float(value)

    frame = pd.DataFrame([values], columns=list(feature_names))
    return float(model.predict(frame)[0])


def extract_stat_series(
    stats_df: pd.DataFrame, player_norm: str, column_candidates: list[str]
):
    stat_col = pick_existing_column(stats_df, column_candidates)
    if stat_col is None:
        return None

    rows = stats_df[stats_df["player_name_norm"] == player_norm].copy()
    if rows.empty:
        return None
    order_cols = [column for column in ["date", "game_id"] if column in rows.columns]
    if order_cols:
        rows = rows.sort_values(order_cols, ascending=True)
    series = pd.to_numeric(rows[stat_col], errors="coerce").dropna()
    if series.empty:
        return None
    return series


def resolve_matchup_date_context(
    home_last_game, away_last_game, game_date_text: str | None
) -> tuple[pd.Timestamp, int, int]:
    if game_date_text:
        try:
            game_date = pd.Timestamp(game_date_text)
        except Exception as exc:
            raise ValueError("Invalid game_date. Use YYYY-MM-DD.") from exc
        if pd.isna(game_date):
            raise ValueError("Invalid game_date. Use YYYY-MM-DD.")
    else:
        game_date = max(home_last_game, away_last_game) + pd.Timedelta(days=1)

    home_rest_days = max((game_date - home_last_game).days, 0)
    away_rest_days = max((game_date - away_last_game).days, 0)
    return game_date, home_rest_days, away_rest_days


def build_matchup_features(
    home_state: dict[str, Any],
    away_state: dict[str, Any],
    home_rest_days: int,
    away_rest_days: int,
) -> pd.DataFrame:
    home_b2b = int(home_rest_days <= 1)
    away_b2b = int(away_rest_days <= 1)

    home_elo_pre = _state_value(home_state, "elo_pre")
    away_elo_pre = _state_value(away_state, "elo_pre")

    home_avg_pts_for_last5 = _state_value(home_state, "avg_pts_for_last5")
    home_avg_pts_against_last5 = _state_value(home_state, "avg_pts_against_last5")
    away_avg_pts_for_last5 = _state_value(away_state, "avg_pts_for_last5")
    away_avg_pts_against_last5 = _state_value(away_state, "avg_pts_against_last5")

    home_win_pct_last5 = _state_value(home_state, "win_pct_last5")
    away_win_pct_last5 = _state_value(away_state, "win_pct_last5")

    home_netrtg_last10 = _state_value(home_state, "netrtg_last10")
    away_netrtg_last10 = _state_value(away_state, "netrtg_last10")

    home_efg_last10 = _state_value(home_state, "efg_last10")
    away_efg_last10 = _state_value(away_state, "efg_last10")
    home_tov_pct_last10 = _state_value(home_state, "tov_pct_last10")
    away_tov_pct_last10 = _state_value(away_state, "tov_pct_last10")
    home_orb_pct_last10 = _state_value(home_state, "orb_pct_last10")
    away_orb_pct_last10 = _state_value(away_state, "orb_pct_last10")
    home_ftr_last10 = _state_value(home_state, "ftr_last10")
    away_ftr_last10 = _state_value(away_state, "ftr_last10")

    home_net_last5 = home_avg_pts_for_last5 - home_avg_pts_against_last5
    away_net_last5 = away_avg_pts_for_last5 - away_avg_pts_against_last5

    row = {
        "home_elo_pre": home_elo_pre,
        "away_elo_pre": away_elo_pre,
        "home_avg_pts_for_last5": home_avg_pts_for_last5,
        "home_avg_pts_against_last5": home_avg_pts_against_last5,
        "away_avg_pts_for_last5": away_avg_pts_for_last5,
        "away_avg_pts_against_last5": away_avg_pts_against_last5,
        "home_win_pct_last5": home_win_pct_last5,
        "away_win_pct_last5": away_win_pct_last5,
        "home_rest_days": float(home_rest_days),
        "away_rest_days": float(away_rest_days),
        "home_b2b": float(home_b2b),
        "away_b2b": float(away_b2b),
        "home_netrtg_last10": home_netrtg_last10,
        "away_netrtg_last10": away_netrtg_last10,
        "home_efg_last10": home_efg_last10,
        "home_tov_pct_last10": home_tov_pct_last10,
        "home_orb_pct_last10": home_orb_pct_last10,
        "home_ftr_last10": home_ftr_last10,
        "away_efg_last10": away_efg_last10,
        "away_tov_pct_last10": away_tov_pct_last10,
        "away_orb_pct_last10": away_orb_pct_last10,
        "away_ftr_last10": away_ftr_last10,
        "net_diff_last5": home_net_last5 - away_net_last5,
        "win_pct_diff_last5": home_win_pct_last5 - away_win_pct_last5,
        "elo_diff": home_elo_pre - away_elo_pre,
        "rest_diff": float(home_rest_days - away_rest_days),
        "b2b_diff": float(home_b2b - away_b2b),
        "netrtg_diff_last10": home_netrtg_last10 - away_netrtg_last10,
        "efg_diff_last10": home_efg_last10 - away_efg_last10,
        "tov_pct_diff_last10": home_tov_pct_last10 - away_tov_pct_last10,
        "orb_pct_diff_last10": home_orb_pct_last10 - away_orb_pct_last10,
        "ftr_diff_last10": home_ftr_last10 - away_ftr_last10,
    }
    return pd.DataFrame([row], columns=FEATURES)
