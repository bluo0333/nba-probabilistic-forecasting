from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.predict import router as predict_router
from app.api.routes.props import router as props_router
from app.api.routes.teams import router as teams_router
from app.api.routes.teams import system_router
from app.core.config import settings
from app.services import model_service

app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    lifespan=model_service.lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system_router)
app.include_router(teams_router)
app.include_router(predict_router)
app.include_router(props_router)
