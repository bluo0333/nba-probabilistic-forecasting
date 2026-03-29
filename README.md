# NBA Probabilistic Forecasting Platform

End-to-end NBA analytics project for:

- game outcome probability modeling
- matchup prediction (CLI + local web app)
- player prop feature/model pipelines
- play-by-play to player-game stat table generation in DuckDB

## Features

### Game Forecasting

- Ingests historical NBA tables from Kaggle SQLite into DuckDB.
- Builds leakage-safe team features:
  - MOV-adjusted Elo with offseason regression
  - rolling form (last 5 games)
  - pace-adjusted net rating (last 10)
  - rolling Four Factors (last 10)
  - rest / back-to-back fatigue features
- Trains calibrated logistic regression with time-series CV and recency weighting.
- Predicts matchup win probabilities via CLI (`predict_matchup.py`).

### Player Props

- Ingests player game logs from `balldontlie` into `data/player_game_stats.csv`.
- Builds rolling player features (`player_features.csv`).
- Trains regression models for:
  - points
  - assists
  - rebounds
  - threes made
- Scores sportsbook lines from `data/player_lines_input.csv` and outputs:
  - predicted mean
  - model probability over
  - sportsbook implied probability
  - edge

### Play-by-Play Player Stat Build (New)

- Builds a DuckDB `player_game_stats` table from play-by-play event data.
- Correctly attributes:
  - points to `player1_id`
  - rebounds to `player1_id`
  - assists to `player2_id` on made field goals
- Includes metadata join (`player_name`, `position`, `team_id`) from `common_player_info` / `player`.
- Deduplicates by `(game_id, eventnum)` before aggregation to prevent double counting.

### Local Web App

- Runs a local WSGI app at `http://127.0.0.1:8000`.
- Provides:
  - team-tile matchup selector
  - today/latest slate display from local data
  - matchup probability output
  - player prop calculator for `points`, `rebounds`, and `3ps`

## Project Structure

`pipelines/`

- `ingest.py`
- `build_features.py`
- `train.py`
- `predict_matchup.py`
- `ingest_balldontlie.py`
- `build_player_features.py`
- `build_player_features_duckdb.py`
- `train_player_model.py`
- `predict_player_props.py`
- `build_player_game_stats.py`

`app/`

- `main.py`

`data/`

- `nba.duckdb`
- `raw/kaggle/` (source datasets)
- generated CSV artifacts for player pipelines

`models/`

- `logistic_model.pkl` (game model)
- player prop models (`points_model.pkl`, `assists_model.pkl`, `rebounds_model.pkl`, `threes_model.pkl`)

`sql/`

- `build_player_features.sql`

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Game Model Workflow

```powershell
python pipelines/ingest.py
python pipelines/build_features.py
python pipelines/train.py
python pipelines/predict_matchup.py --home IND --away LAL
```

Start web app:

```powershell
python app/main.py
```

## Player Props Workflow (balldontlie)

Optional API key:

```powershell
$env:BALLDONTLIE_API_KEY="your_key_here"
```

Run pipeline:

```powershell
python pipelines/ingest_balldontlie.py
python pipelines/build_player_features.py
python pipelines/train_player_model.py
python pipelines/predict_player_props.py
```

Input lines file schema (`data/player_lines_input.csv`):

- `player`
- `stat` (`points`, `assists`, `rebounds`, `3ps`, `pra`)
- `line`
- `over_odds`
- `under_odds`

Output:

- `data/player_prop_predictions.csv`

## Build `player_game_stats` from Kaggle Play-by-Play

`build_player_game_stats.py` expects these DuckDB tables:

- `play_by_play`
- `player`
- `common_player_info`

If they are not yet in `data/nba.duckdb`, load them from CSV first:

```powershell
@'
import duckdb
from pathlib import Path

db = Path("data/nba.duckdb")
csv_root = Path("data/raw/kaggle/csv")
con = duckdb.connect(str(db))
con.execute("CREATE OR REPLACE TABLE play_by_play AS SELECT * FROM read_csv_auto(?, header=true)", [str(csv_root / "play_by_play.csv")])
con.execute("CREATE OR REPLACE TABLE player AS SELECT * FROM read_csv_auto(?, header=true)", [str(csv_root / "player.csv")])
con.execute("CREATE OR REPLACE TABLE common_player_info AS SELECT * FROM read_csv_auto(?, header=true)", [str(csv_root / "common_player_info.csv")])
con.close()
'@ | python -
```

Then build:

```powershell
python pipelines/build_player_game_stats.py
```

## Build Leakage-Safe `player_features` in DuckDB

From `player_game_stats`, create a model-ready `player_features` table with
rolling/expanding windows that exclude the current game:

```powershell
python pipelines/build_player_features_duckdb.py
```

## Current Out-of-Sample Game Model Performance (2018-2023)

- Log Loss: `0.633`
- Brier Score: `0.221`
- Accuracy: `64.4%`

Baseline (always pick home team): `56.5%` accuracy.
