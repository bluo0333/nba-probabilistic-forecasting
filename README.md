# NBA Probabilistic Forecasting Platform

An end-to-end NBA game outcome forecasting system that ingests historical league data (1946–present),
engineers rolling team-level features, trains calibrated probabilistic models,
and serves win probability forecasts via a lightweight web application.

## Goals
- Build a production-style data pipeline for large-scale sports data
- Forecast NBA game outcomes using probabilistic models (not binary predictions)
- Evaluate models using backtesting, calibration, and proper scoring rules
- Deploy results in a lightweight, reproducible web application

## Data
- Historical NBA games, teams, players, box scores, and play-by-play data
- Source: Kaggle NBA relational database (updated daily)

## Project Structure
- `pipelines/`: data ingestion, feature engineering, model training, and backtesting
- `sql/`: schema definitions, transformations, and analytical queries
- `app/`: web interface for viewing forecasts and evaluation results

## Status
Project in progress - currently implementing data ingestion and feature engineering pipelines.
