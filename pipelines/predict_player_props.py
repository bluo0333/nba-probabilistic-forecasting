from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm


REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = REPO_ROOT / "data" / "player_features.csv"
PLAYER_STATS_PATH = REPO_ROOT / "data" / "player_game_stats.csv"
LINES_INPUT_PATH = REPO_ROOT / "data" / "player_lines_input.csv"
OUTPUT_PATH = REPO_ROOT / "data" / "player_prop_predictions.csv"

FEATURE_COLUMNS = [
    "rolling_points_10",
    "rolling_assists_10",
    "rolling_rebounds_10",
    "rolling_minutes_10",
]

MODEL_PATHS = {
    "points": REPO_ROOT / "models" / "points_model.pkl",
    "assists": REPO_ROOT / "models" / "assists_model.pkl",
    "rebounds": REPO_ROOT / "models" / "rebounds_model.pkl",
}

OUTPUT_COLUMNS = [
    "player",
    "stat",
    "line",
    "predicted_mean",
    "model_probability_over",
    "sportsbook_probability",
    "edge",
]


def normalize_player_name(name: Any) -> str:
    """Normalize player names for robust matching."""
    return str(name or "").strip().lower()


def american_to_implied_probability(odds: float) -> float:
    """Convert American odds to implied probability."""
    if odds == 0:
        raise ValueError("American odds cannot be zero.")
    if odds < 0:
        return -odds / (-odds + 100.0)
    return 100.0 / (odds + 100.0)


def load_models() -> dict[str, Any]:
    """Load trained regression models from disk."""
    models: dict[str, Any] = {}
    for stat, path in MODEL_PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        models[stat] = joblib.load(path)
    return models


def load_player_features(path: Path) -> pd.DataFrame:
    """Load player feature rows and validate required columns."""
    if not path.is_file():
        raise FileNotFoundError(f"Feature file not found: {path}")
    df = pd.read_csv(path)

    required = {"player_name", "date"} | set(FEATURE_COLUMNS)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["player_name_norm"] = df["player_name"].map(normalize_player_name)
    return df


def load_lines_input(path: Path) -> pd.DataFrame:
    """Load user sportsbook lines."""
    if not path.is_file():
        raise FileNotFoundError(f"Input lines file not found: {path}")

    df = pd.read_csv(path)
    required = {"player", "stat", "line", "over_odds", "under_odds"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    return df


def get_latest_feature_row(features_df: pd.DataFrame, player_name: str) -> pd.Series | None:
    """Return latest available feature row for a player."""
    norm_name = normalize_player_name(player_name)
    player_rows = features_df[features_df["player_name_norm"] == norm_name].copy()
    if player_rows.empty:
        return None
    player_rows = player_rows.sort_values(["date", "game_id"], ascending=[False, False])
    return player_rows.iloc[0]


def build_std_tables(stats_df: pd.DataFrame) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Build player-level and global std-dev lookup tables for stats."""
    frame = stats_df.copy()
    frame["player_name_norm"] = frame["player_name"].map(normalize_player_name)
    frame["pra"] = frame["points"] + frame["assists"] + frame["rebounds"]

    player_std: dict[str, dict[str, float]] = {}
    for stat in ["points", "assists", "rebounds", "pra"]:
        std_series = frame.groupby("player_name_norm")[stat].std(ddof=1)
        player_std[stat] = std_series.to_dict()

    global_std = {
        "points": float(frame["points"].std(ddof=1)),
        "assists": float(frame["assists"].std(ddof=1)),
        "rebounds": float(frame["rebounds"].std(ddof=1)),
        "pra": float(frame["pra"].std(ddof=1)),
    }
    return player_std, global_std


def resolve_std(
    player_name: str,
    stat: str,
    player_std: dict[str, dict[str, float]],
    global_std: dict[str, float],
) -> float:
    """Resolve std dev for a player/stat with global fallback."""
    norm_name = normalize_player_name(player_name)
    std_value = player_std.get(stat, {}).get(norm_name)
    if std_value is None or not np.isfinite(std_value) or std_value <= 0:
        std_value = global_std.get(stat)
    if std_value is None or not np.isfinite(std_value) or std_value <= 0:
        std_value = 1.0
    return float(std_value)


def predict_stat_mean(models: dict[str, Any], features: pd.DataFrame, stat: str) -> float:
    """Predict expected value for one stat type."""
    if stat in {"points", "assists", "rebounds"}:
        return float(models[stat].predict(features)[0])
    if stat == "pra":
        pts = float(models["points"].predict(features)[0])
        ast = float(models["assists"].predict(features)[0])
        reb = float(models["rebounds"].predict(features)[0])
        return pts + ast + reb
    raise ValueError(f"Unsupported stat '{stat}'. Expected points/assists/rebounds/pra.")


def load_stats_for_std(path: Path) -> pd.DataFrame:
    """Load historical player stats used for variance estimation."""
    if not path.is_file():
        raise FileNotFoundError(f"Player stats file not found: {path}")
    df = pd.read_csv(path)
    required = {"player_name", "points", "assists", "rebounds"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    return df


def main() -> None:
    models = load_models()
    features_df = load_player_features(FEATURES_PATH)
    lines_df = load_lines_input(LINES_INPUT_PATH)
    stats_df = load_stats_for_std(PLAYER_STATS_PATH)
    player_std, global_std = build_std_tables(stats_df)

    results: list[dict[str, Any]] = []

    for idx, row in lines_df.iterrows():
        player_name = str(row.get("player", "")).strip()
        stat = str(row.get("stat", "")).strip().lower()

        try:
            line = float(row.get("line"))
            over_odds = float(row.get("over_odds"))
        except (TypeError, ValueError):
            print(f"Skipping row {idx}: invalid line/odds values.")
            continue

        if stat not in {"points", "assists", "rebounds", "pra"}:
            print(f"Skipping row {idx}: unsupported stat '{stat}'.")
            continue

        feature_row = get_latest_feature_row(features_df, player_name)
        if feature_row is None:
            print(f"Skipping row {idx}: player '{player_name}' not found in feature data.")
            continue

        feature_values = pd.DataFrame([feature_row[FEATURE_COLUMNS].to_dict()], columns=FEATURE_COLUMNS)
        if feature_values.isnull().any(axis=None):
            print(f"Skipping row {idx}: missing feature values for '{player_name}'.")
            continue

        try:
            predicted_mean = predict_stat_mean(models, feature_values, stat)
            std_dev = resolve_std(player_name, stat, player_std, global_std)
            model_probability_over = float(1.0 - norm.cdf(line, loc=predicted_mean, scale=std_dev))
            sportsbook_probability = float(american_to_implied_probability(over_odds))
        except Exception as exc:
            print(f"Skipping row {idx}: could not score '{player_name}' {stat}. Error: {exc}")
            continue

        edge = model_probability_over - sportsbook_probability
        results.append(
            {
                "player": player_name,
                "stat": stat,
                "line": line,
                "predicted_mean": predicted_mean,
                "model_probability_over": model_probability_over,
                "sportsbook_probability": sportsbook_probability,
                "edge": edge,
            }
        )

    output_df = pd.DataFrame(results, columns=OUTPUT_COLUMNS)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(output_df)} predictions to {OUTPUT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: prop prediction pipeline failed: {exc}")
        raise SystemExit(1)
