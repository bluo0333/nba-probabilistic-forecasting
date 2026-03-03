# NBA Probabilistic Forecasting Platform

This project builds an NBA game outcome forecasting system using historical league data (1946 - present). It ingests raw relational data, engineers team level features, trains probabilistic models, and produces win probability estimates for matchups.

The focus is on building a reproducible analytics pipeline.

## Modeling Approach

The current model combines:

- **Margin-of-victory adjusted Elo**
  - Home court advantage
  - Offseason regression
- **Rolling team form (last 5 games)**
  - Pace adjusted offensive & defensive efficiency
  - Net rating differentials
  - Win percentage differentials
- **Fatigue features**
  - Rest days
  - Back-to-back indicators
  - Rest differential

All features computed strictly using past information to avoid leakage.

## Current Out of Sample Performance (2018 - 2023)

- **Log Loss:** 0.633  
- **Brier Score:** 0.221  
- **Accuracy:** 64.4%

Baseline (always picking the home team):
- Accuracy: 56.5%

The model improves winner prediction by ~8 percentage points over a naive strategy.

## Project Structure

pipelines/
- ingest.py  
- build_features.py  
- train.py  

app/
- main.py  

models/
- logistic_model.pkl  

data/
- nba.duckdb  

## Status

Core modeling pipeline complete.  
Plan to expose predictions via API and build a lightweight web interface.
