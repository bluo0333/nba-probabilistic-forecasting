from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

PROCESSED_GAMES_PATH = DATA_DIR / "processed_games.csv"
FINAL_FEATURES_PATH = DATA_DIR / "final_features.csv"
TEAMS_PATH = DATA_DIR / "teams.csv"
TEAM_STATES_PATH = DATA_DIR / "team_states.csv"
PLAYER_FEATURES_PATH = DATA_DIR / "player_features.csv"
PLAYER_STATS_PATH = DATA_DIR / "player_game_stats.csv"
MATCHUP_MODEL_PATH = MODELS_DIR / "logistic_model.pkl"
PLAYER_PROP_MODEL_PATHS = {
    "points": MODELS_DIR / "points_model.pkl",
    "rebounds": MODELS_DIR / "rebounds_model.pkl",
    "3ps": MODELS_DIR / "threes_model.pkl",
}


@dataclass(frozen=True)
class Settings:
    app_title: str = "NBA Probabilistic Forecasting API"
    app_version: str = "1.0.0"
    cors_allow_origins: tuple[str, ...] = ("*",)


settings = Settings()
