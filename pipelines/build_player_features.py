from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "player_game_stats.csv"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "player_features.csv"


def load_player_stats(path: Path) -> pd.DataFrame:
    """Load player game stats and validate required columns."""
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)
    required_columns = {
        "player_id",
        "player_name",
        "game_id",
        "date",
        "minutes",
        "points",
        "assists",
        "rebounds",
        "team_id",
        "opponent_team_id",
    }
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling stat features per player using prior games only."""
    df = df.sort_values(["player_id", "date", "game_id"]).copy()
    grouped = df.groupby("player_id", group_keys=False)

    df["rolling_points_5"] = grouped["points"].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )
    df["rolling_points_10"] = grouped["points"].transform(
        lambda s: s.shift(1).rolling(window=10, min_periods=1).mean()
    )
    df["rolling_assists_10"] = grouped["assists"].transform(
        lambda s: s.shift(1).rolling(window=10, min_periods=1).mean()
    )
    df["rolling_rebounds_10"] = grouped["rebounds"].transform(
        lambda s: s.shift(1).rolling(window=10, min_periods=1).mean()
    )
    df["rolling_minutes_10"] = grouped["minutes"].transform(
        lambda s: s.shift(1).rolling(window=10, min_periods=1).mean()
    )
    df["pra"] = df["points"] + df["assists"] + df["rebounds"]
    return df


def main() -> None:
    print(f"Loading player stats from {INPUT_PATH}...")
    df = load_player_stats(INPUT_PATH)
    print("Building rolling player features...")
    feature_df = add_rolling_features(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(feature_df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: feature build failed: {exc}")
        raise SystemExit(1)
