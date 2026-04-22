from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.db.connection import get_connection
from app.services import data_service, feature_service

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MATCHUP_MODEL_PATH = PROJECT_ROOT / "models" / "logistic_model.pkl"

PLAYER_FEATURES_PATH = PROJECT_ROOT / "data" / "player_features.csv"
PLAYER_STATS_PATH = PROJECT_ROOT / "data" / "player_game_stats.csv"
COMMON_PLAYER_INFO_PATH = (
    PROJECT_ROOT / "data" / "raw" / "kaggle" / "csv" / "common_player_info.csv"
)

PLAYER_PROP_MODEL_PATHS = {
    "points": PROJECT_ROOT / "models" / "points_model.pkl",
    "rebounds": PROJECT_ROOT / "models" / "rebounds_model.pkl",
    "3ps": PROJECT_ROOT / "models" / "threes_model.pkl",
}
PLAYER_PROP_MEAN_COLUMN_CANDIDATES = {
    "points": ["rolling_points_10", "points"],
    "rebounds": ["rolling_rebounds_10", "rebounds"],
    "3ps": [
        "rolling_3pm_10",
        "rolling_fg3m_10",
        "rolling_threes_10",
        "threes_made",
        "fg3m",
    ],
}
PLAYER_PROP_STAT_COLUMN_CANDIDATES = {
    "points": ["points", "pts"],
    "rebounds": ["rebounds", "reb", "trb"],
    "3ps": ["threes_made", "fg3m", "three_pointers_made", "three_pointers"],
}
PLAYER_PROP_DEFAULT_STD = {"points": 6.0, "rebounds": 3.0, "3ps": 1.5}
PLAYER_PROP_TYPES = ("points", "rebounds", "3ps")
PLAYER_PROP_LABELS = {"points": "Points", "rebounds": "Rebounds", "3ps": "3PM"}

_MATCHUP_MODEL: Any | None = None
_PLAYER_PROP_CONTEXT: dict[str, Any] | None = None


def load_matchup_model() -> Any:
    global _MATCHUP_MODEL
    if _MATCHUP_MODEL is not None:
        return _MATCHUP_MODEL
    if not MATCHUP_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Matchup model not found: {MATCHUP_MODEL_PATH}")
    _MATCHUP_MODEL = joblib.load(MATCHUP_MODEL_PATH)
    return _MATCHUP_MODEL


def predict_matchup(
    home: str, away: str, game_date_text: str | None = None
) -> dict[str, Any]:
    model = load_matchup_model()

    with get_connection() as conn:
        home_id = data_service.resolve_team_id(conn, home)
        away_id = data_service.resolve_team_id(conn, away)
        if home_id == away_id:
            raise ValueError("Home and away teams must be different.")

        home_meta, away_meta = data_service.get_team_metadata(conn, home_id, away_id)

        home_state = data_service.get_latest_team_state(conn, home_id)
        away_state = data_service.get_latest_team_state(conn, away_id)
        home_last_game = data_service.get_last_game_date(conn, home_id)
        away_last_game = data_service.get_last_game_date(conn, away_id)

    game_date, home_rest_days, away_rest_days = (
        feature_service.resolve_matchup_date_context(
            home_last_game,
            away_last_game,
            game_date_text,
        )
    )
    features = feature_service.build_matchup_features(
        home_state, away_state, home_rest_days, away_rest_days
    )

    try:
        home_win_prob = float(model.predict_proba(features)[:, 1][0])
    except Exception as exc:
        raise RuntimeError(f"Model inference failed: {exc}") from exc

    away_win_prob = 1.0 - home_win_prob
    predicted_winner = (
        home_meta["full_name"] if home_win_prob >= 0.5 else away_meta["full_name"]
    )
    return {
        "matchup": f"{away_meta['abbreviation']} @ {home_meta['abbreviation']}",
        "home_team": home_meta["full_name"],
        "away_team": away_meta["full_name"],
        "home_win_probability": home_win_prob,
        "away_win_probability": away_win_prob,
        "predicted_winner": predicted_winner,
    }


def _load_player_prop_context() -> dict[str, Any]:
    global _PLAYER_PROP_CONTEXT
    if _PLAYER_PROP_CONTEXT is not None:
        return _PLAYER_PROP_CONTEXT

    features_df: pd.DataFrame | None = None
    stats_df: pd.DataFrame | None = None
    models: dict[str, Any] = {}
    player_names: set[str] = set()

    if PLAYER_FEATURES_PATH.is_file():
        features_df = pd.read_csv(PLAYER_FEATURES_PATH)
        if "player_name" in features_df.columns:
            features_df["player_name"] = (
                features_df["player_name"].astype(str).str.strip()
            )
            features_df["player_name_norm"] = features_df["player_name"].map(
                feature_service.normalize_name
            )
            player_names.update(features_df["player_name"].dropna().tolist())
        if "date" in features_df.columns:
            features_df["date"] = pd.to_datetime(features_df["date"], errors="coerce")

    if PLAYER_STATS_PATH.is_file():
        stats_df = pd.read_csv(PLAYER_STATS_PATH)
        if "player_name" in stats_df.columns:
            stats_df["player_name"] = stats_df["player_name"].astype(str).str.strip()
            stats_df["player_name_norm"] = stats_df["player_name"].map(
                feature_service.normalize_name
            )
            player_names.update(stats_df["player_name"].dropna().tolist())
        if "date" in stats_df.columns:
            stats_df["date"] = pd.to_datetime(stats_df["date"], errors="coerce")

    if not player_names and COMMON_PLAYER_INFO_PATH.is_file():
        info = pd.read_csv(COMMON_PLAYER_INFO_PATH, usecols=["display_first_last"])
        values = info["display_first_last"].dropna().astype(str).str.strip()
        player_names.update([name for name in values if name])

    for prop_type, model_path in PLAYER_PROP_MODEL_PATHS.items():
        if model_path.is_file():
            try:
                models[prop_type] = joblib.load(model_path)
            except Exception:
                continue

    players = sorted(player_names)
    name_map = {feature_service.normalize_name(name): name for name in players}
    _PLAYER_PROP_CONTEXT = {
        "features_df": features_df,
        "stats_df": stats_df,
        "models": models,
        "players": players,
        "name_map": name_map,
    }
    return _PLAYER_PROP_CONTEXT


def get_player_names() -> list[str]:
    return _load_player_prop_context()["players"]


def predict_player_prop(
    player: str, prop_type: str, side: str, line: float, odds: float
) -> dict[str, Any]:
    context = _load_player_prop_context()
    prop_type_norm = str(prop_type).strip().lower()
    side_norm = str(side).strip().lower()

    if prop_type_norm not in PLAYER_PROP_TYPES:
        raise ValueError("Line type must be one of: points, rebounds, 3ps.")
    if side_norm not in {"over", "under"}:
        raise ValueError("Side must be over or under.")

    player_norm = feature_service.normalize_name(player)
    canonical_name = context["name_map"].get(player_norm)
    if not canonical_name:
        raise ValueError(f"Player '{player}' not found.")

    features_df = context["features_df"]
    stats_df = context["stats_df"]
    models = context["models"]

    predicted_mean: float | None = None
    mean_source = ""

    if features_df is not None:
        latest_row = feature_service.latest_feature_row(features_df, player_norm)
        if latest_row is not None:
            model = models.get(prop_type_norm)
            if model is not None:
                predicted_mean = feature_service.predict_mean_from_model(
                    model, latest_row
                )
                if predicted_mean is not None:
                    mean_source = "model"
            if predicted_mean is None:
                mean_col = feature_service.pick_existing_column(
                    features_df,
                    PLAYER_PROP_MEAN_COLUMN_CANDIDATES[prop_type_norm],
                )
                if mean_col is not None:
                    value = latest_row.get(mean_col)
                    if value is not None and not pd.isna(value):
                        predicted_mean = float(value)
                        mean_source = f"rolling ({mean_col})"

    player_series = None
    if stats_df is not None:
        player_series = feature_service.extract_stat_series(
            stats_df,
            player_norm,
            PLAYER_PROP_STAT_COLUMN_CANDIDATES[prop_type_norm],
        )
    if predicted_mean is None and player_series is not None and len(player_series) > 0:
        predicted_mean = float(player_series.tail(10).mean())
        mean_source = "recent average (last 10)"

    if predicted_mean is None:
        raise ValueError(
            "Could not compute this player prop yet. Missing player feature/stat data. "
            "Run player ingestion/build/train pipelines first."
        )

    std_dev: float | None = None
    if player_series is not None and len(player_series) >= 2:
        std_dev = float(player_series.std(ddof=1))
    if (
        std_dev is None or not math.isfinite(std_dev) or std_dev <= 0
    ) and stats_df is not None:
        global_col = feature_service.pick_existing_column(
            stats_df,
            PLAYER_PROP_STAT_COLUMN_CANDIDATES[prop_type_norm],
        )
        if global_col is not None:
            global_std = float(
                pd.to_numeric(stats_df[global_col], errors="coerce")
                .dropna()
                .std(ddof=1)
            )
            if math.isfinite(global_std) and global_std > 0:
                std_dev = global_std

    if std_dev is None or not math.isfinite(std_dev) or std_dev <= 0:
        std_dev = PLAYER_PROP_DEFAULT_STD[prop_type_norm]

    prob_over = float(1.0 - feature_service.normal_cdf(line, predicted_mean, std_dev))
    hit_probability = prob_over if side_norm == "over" else 1.0 - prob_over
    implied_probability = float(feature_service.american_to_implied_probability(odds))
    edge = hit_probability - implied_probability

    return {
        "player": canonical_name,
        "line_type": prop_type_norm,
        "line_type_label": PLAYER_PROP_LABELS[prop_type_norm],
        "side": side_norm,
        "line": float(line),
        "odds": float(odds),
        "predicted_mean": predicted_mean,
        "std_dev": std_dev,
        "hit_probability": hit_probability,
        "implied_probability": implied_probability,
        "edge": edge,
        "mean_source": mean_source or "fallback",
    }
