from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.predict import router as predict_router
from app.routes.props import router as props_router
from app.routes.teams import router as teams_router


@asynccontextmanager
async def lifespan(_app: FastAPI):
    from app.services import data_service, model_service

    model_service.load_matchup_model()
    data_service.preload_prediction_data()
    yield

app = FastAPI(
    title="NBA Probabilistic Forecasting API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(teams_router)
app.include_router(predict_router)
app.include_router(props_router)


@app.get("/health", tags=["health"])
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "NBA Prediction API running"}
