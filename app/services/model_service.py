from __future__ import annotations

import logging
import math
import time
from contextlib import asynccontextmanager
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI

from app.core.config import MATCHUP_MODEL_PATH, PLAYER_FEATURES_PATH, PLAYER_PROP_MODEL_PATHS, PLAYER_STATS_PATH
from app.services import data_service
from app.utils import feature_utils

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
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    warmup_models()
    yield


def warmup_models() -> None:
    load_matchup_model()
    data_service.preload_team_index()
    data_service.preload_prediction_data()


def load_matchup_model() -> Any:
    global _MATCHUP_MODEL
    if _MATCHUP_MODEL is not None:
        return _MATCHUP_MODEL
    if not MATCHUP_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Matchup model not found: {MATCHUP_MODEL_PATH}")
    _MATCHUP_MODEL = joblib.load(MATCHUP_MODEL_PATH)
    return _MATCHUP_MODEL


def predict_matchup(
    home: str,
    away: str,
    game_date_text: str | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    logger.info(
        "predict_matchup start home=%s away=%s game_date=%s",
        home,
        away,
        game_date_text,
    )
    try:
        model = load_matchup_model()

        home_id = data_service.resolve_team_id(home)
        away_id = data_service.resolve_team_id(away)
        if home_id == away_id:
            raise ValueError("Home and away teams must be different.")

        home_meta, away_meta = data_service.get_team_metadata(home_id, away_id)
        home_state, away_state, home_last_game, away_last_game = data_service.get_matchup_snapshot(
            home_id=home_id,
            away_id=away_id,
        )

        _, home_rest_days, away_rest_days = feature_utils.resolve_matchup_date_context(
            home_last_game,
            away_last_game,
            game_date_text,
        )
        features = feature_utils.build_matchup_features(
            home_state,
            away_state,
            home_rest_days,
            away_rest_days,
        )
        home_win_prob = float(model.predict_proba(features)[:, 1][0])
    except ValueError:
        raise
    except FileNotFoundError:
        raise
    except Exception as exc:
        logger.exception(
            "predict_matchup failed home=%s away=%s game_date=%s",
            home,
            away,
            game_date_text,
        )
        raise RuntimeError(f"Model inference failed: {exc}") from exc

    away_win_prob = 1.0 - home_win_prob
    predicted_winner = (
        home_meta["full_name"] if home_win_prob >= 0.5 else away_meta["full_name"]
    )
    response = {
        "matchup": f"{away_meta['abbreviation']} @ {home_meta['abbreviation']}",
        "home_team": home_meta["full_name"],
        "away_team": away_meta["full_name"],
        "home_win_probability": home_win_prob,
        "away_win_probability": away_win_prob,
        "predicted_winner": predicted_winner,
    }
    logger.info(
        "predict_matchup end home=%s away=%s elapsed_ms=%.2f",
        home,
        away,
        (time.perf_counter() - started_at) * 1000.0,
    )
    return response


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
                feature_utils.normalize_name
            )
            player_names.update(features_df["player_name"].dropna().tolist())
        if "date" in features_df.columns:
            features_df["date"] = pd.to_datetime(features_df["date"], errors="coerce")

    if PLAYER_STATS_PATH.is_file():
        stats_df = pd.read_csv(PLAYER_STATS_PATH)
        if "player_name" in stats_df.columns:
            stats_df["player_name"] = stats_df["player_name"].astype(str).str.strip()
            stats_df["player_name_norm"] = stats_df["player_name"].map(
                feature_utils.normalize_name
            )
            player_names.update(stats_df["player_name"].dropna().tolist())
        if "date" in stats_df.columns:
            stats_df["date"] = pd.to_datetime(stats_df["date"], errors="coerce")

    for prop_type, model_path in PLAYER_PROP_MODEL_PATHS.items():
        if model_path.is_file():
            try:
                models[prop_type] = joblib.load(model_path)
            except Exception:
                logger.warning("failed_loading_player_prop_model path=%s", model_path)

    players = sorted(player_names)
    name_map = {feature_utils.normalize_name(name): name for name in players}
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


def get_player_recent_games(
    player: str,
    prop_type: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    context = _load_player_prop_context()
    prop_type_norm = str(prop_type).strip().lower()
    if prop_type_norm not in PLAYER_PROP_TYPES:
        raise ValueError("Line type must be one of: points, rebounds, 3ps.")

    player_norm = feature_utils.normalize_name(player)
    canonical_name = context["name_map"].get(player_norm)
    if not canonical_name:
        raise ValueError(f"Player '{player}' not found.")

    stats_df = context["stats_df"]
    if stats_df is None or stats_df.empty:
        return []

    stat_column = feature_utils.pick_existing_column(
        stats_df,
        PLAYER_PROP_STAT_COLUMN_CANDIDATES[prop_type_norm],
    )
    if stat_column is None:
        return []

    player_games = stats_df[stats_df["player_name_norm"] == player_norm].copy()
    if player_games.empty:
        return []

    player_games = player_games.sort_values(["date", "game_id"], ascending=False).head(
        max(1, min(int(limit), 50))
    )

    rows: list[dict[str, Any]] = []
    for row in player_games.to_dict("records"):
        rows.append(
            {
                "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                "opponent": data_service.team_name_for_id(row.get("opponent_team_id")),
                "value": float(row.get(stat_column) or 0.0),
                "mins": float(row.get("minutes") or 0.0),
            }
        )
    return rows


def predict_player_prop(
    player: str,
    prop_type: str,
    side: str,
    line: float,
    odds: float,
) -> dict[str, Any]:
    context = _load_player_prop_context()
    prop_type_norm = str(prop_type).strip().lower()
    side_norm = str(side).strip().lower()

    if prop_type_norm not in PLAYER_PROP_TYPES:
        raise ValueError("Line type must be one of: points, rebounds, 3ps.")
    if side_norm not in {"over", "under"}:
        raise ValueError("Side must be over or under.")

    player_norm = feature_utils.normalize_name(player)
    canonical_name = context["name_map"].get(player_norm)
    if not canonical_name:
        raise ValueError(f"Player '{player}' not found.")

    features_df = context["features_df"]
    stats_df = context["stats_df"]
    models = context["models"]

    predicted_mean: float | None = None
    mean_source = ""

    if features_df is not None:
        latest_row = feature_utils.latest_feature_row(features_df, player_norm)
        if latest_row is not None:
            model = models.get(prop_type_norm)
            if model is not None:
                predicted_mean = feature_utils.predict_mean_from_model(model, latest_row)
                if predicted_mean is not None:
                    mean_source = "model"
            if predicted_mean is None:
                mean_col = feature_utils.pick_existing_column(
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
        player_series = feature_utils.extract_stat_series(
            stats_df,
            player_norm,
            PLAYER_PROP_STAT_COLUMN_CANDIDATES[prop_type_norm],
        )
    if predicted_mean is None and player_series is not None and len(player_series) > 0:
        predicted_mean = float(player_series.tail(10).mean())
        mean_source = "recent average (last 10)"

    if predicted_mean is None:
        raise ValueError(
            "Could not compute this player prop yet. Missing player feature/stat data."
        )

    std_dev: float | None = None
    if player_series is not None and len(player_series) >= 2:
        std_dev = float(player_series.std(ddof=1))
    if (
        std_dev is None or not math.isfinite(std_dev) or std_dev <= 0
    ) and stats_df is not None:
        global_col = feature_utils.pick_existing_column(
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

    prob_over = float(1.0 - feature_utils.normal_cdf(line, predicted_mean, std_dev))
    hit_probability = prob_over if side_norm == "over" else 1.0 - prob_over
    implied_probability = float(feature_utils.american_to_implied_probability(odds))
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
