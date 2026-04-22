# NBA Probabilistic Forecasting Platform

## Overview

Full-stack sports analytics platform that predicts NBA game outcomes and player prop performance using historical data, engineered features, and machine learning models.

The system includes:

- a FastAPI backend serving real-time predictions via a REST API
- a React frontend for interactive matchup analysis
- a DuckDB-based data pipeline for ingestion, feature engineering, and model training

The project is designed with production-style architecture, separating data pipelines, model inference, and API layers into modular components.

---

## Demo

### API (Swagger UI)

![Swagger UI](assets/swagger.png)

Interactive frontend for selecting matchups and viewing model predictions.

### Main Interface

![Main UI](assets/main.png)

### Team Selection UI

![Team Selector](assets/team-selector.png)

---

## Features

## Tech Stack

- Python 3
- FastAPI
- DuckDB
- scikit-learn + joblib
- pandas + numpy + scipy
- React + Vite

## Architecture

```text
                +----------------------+
                |   React Frontend     |
                |   (Vite, fetch API)  |
                +----------+-----------+
                           |
                           | HTTP (JSON)
                           v
                +----------+-----------+
                |      FastAPI App     |
                |      app/main.py     |
                +----+----------+------+
                     |          |
          +----------+          +------------------+
          v                                     v
+---------------------+              +----------------------+
|   Route Layer       |              |   Schema Layer       |
| app/routes/*.py     |              | app/schemas/*.py     |
+----------+----------+              +----------------------+
           |
           v
+----------+----------+
|   Service Layer     |
| app/services/*.py   |
| - data_service      |
| - feature_service   |
| - model_service     |
+----------+----------+
           |
           v
+----------+----------+      +-------------------------+
| DuckDB (data/*.duckdb)|    | Models (models/*.pkl)   |
| SQL + historical data |    | sklearn/joblib artifacts|
+----------------------+      +-------------------------+

Pipelines (pipelines/*.py) feed DuckDB tables and model artifacts.
```

## Features

- Matchup probability API:
  - `POST /predict/matchup`
  - `GET /predict/quick?home=...&away=...`
- Teams API:
  - `GET /teams/`
- Player props API:
  - `GET /props/players`
  - `POST /props/predict`
- Historical ingestion and feature pipelines:
  - `pipelines/ingest.py`
  - `pipelines/build_features.py`
  - `pipelines/train.py`
- Player workflow pipelines:
  - `pipelines/ingest_balldontlie.py`
  - `pipelines/build_player_features.py`
  - `pipelines/train_player_model.py`
  - `pipelines/predict_player_props.py`
- Play-by-play aggregation:
  - `pipelines/build_player_game_stats.py`

## Model Performance

Current out-of-sample game model performance (2018-2023):

- Log Loss: `0.633`
- Brier Score: `0.221`
- Accuracy: `64.4%`

Baseline (always pick home team): `56.5%` accuracy.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Backend

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 10000
```

## Run Frontend

```powershell
cd frontend
copy .env.example .env
npm install
npm run dev
```

Frontend environment variable:

```env
VITE_API_BASE=http://localhost:8000
```

## API Endpoints

```text
GET  /health
GET  /teams/
GET  /predict/quick?home=...&away=...
POST /predict/matchup
GET  /props/players
POST /props/predict
```
