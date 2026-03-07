from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


FEATURES_PATH = Path(__file__).resolve().parents[1] / "data" / "player_features.csv"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

FEATURE_COLUMNS = [
    "rolling_points_10",
    "rolling_assists_10",
    "rolling_rebounds_10",
    "rolling_minutes_10",
]

TARGET_TO_MODEL_PATH = {
    "points": MODELS_DIR / "points_model.pkl",
    "assists": MODELS_DIR / "assists_model.pkl",
    "rebounds": MODELS_DIR / "rebounds_model.pkl",
}


def load_features(path: Path) -> pd.DataFrame:
    """Load feature table and validate columns."""
    if not path.is_file():
        raise FileNotFoundError(f"Feature file not found: {path}")

    df = pd.read_csv(path)
    required = set(FEATURE_COLUMNS) | set(TARGET_TO_MODEL_PATH.keys())
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    return df


def train_one_model(df: pd.DataFrame, target: str) -> LinearRegression:
    """Train one linear regression model for the requested target stat."""
    clean = df[FEATURE_COLUMNS + [target]].dropna()
    if clean.empty:
        raise ValueError(f"No training rows available for target '{target}'")

    model = LinearRegression()
    model.fit(clean[FEATURE_COLUMNS], clean[target])

    preds = model.predict(clean[FEATURE_COLUMNS])
    mae = mean_absolute_error(clean[target], preds)
    print(f"Trained {target} model on {len(clean)} rows | MAE={mae:.4f}")
    return model


def main() -> None:
    print(f"Loading features from {FEATURES_PATH}...")
    df = load_features(FEATURES_PATH)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for target, model_path in TARGET_TO_MODEL_PATH.items():
        model = train_one_model(df, target)
        joblib.dump(model, model_path)
        print(f"Saved model: {model_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: training failed: {exc}")
        raise SystemExit(1)
