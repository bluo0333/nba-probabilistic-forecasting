from __future__ import annotations

import math
from typing import Any

import pandas as pd

from pipelines import predict_matchup as matchup_pipeline


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
    return matchup_pipeline.build_feature_row(
        home_state, away_state, home_rest_days, away_rest_days
    )
