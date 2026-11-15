"""Containerised FastAPI inference service (deployed to Kubernetes).

/health  -> liveness  (process is up)
/ready   -> readiness (model loaded; 503 otherwise so the Pod is removed from Service endpoints)
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger("inference")


class PredictRequest(BaseModel):
    features: list[float] = Field(min_length=1, max_length=200)


class PredictResponse(BaseModel):
    prediction: int
    probabilities: list[float]
    model_version: str


def create_app(model_path: str | None = None) -> FastAPI:
    state: dict = {"model": None, "error": None}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        path = model_path or os.getenv("MODEL_PATH", "models/model.joblib")
        try:
            state["model"] = joblib.load(path)
            log.info("model loaded from %s", path)
        except Exception as exc:  # noqa: BLE001
            state["error"] = f"{type(exc).__name__}: cannot load model from {path}"
            log.error("MODEL LOAD FAILED: %s", state["error"])
        yield

    app = FastAPI(title="ML Inference Service", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "alive"}

    @app.get("/ready")
    def ready() -> dict:
        if state["model"] is None:
            raise HTTPException(status_code=503, detail=state["error"] or "model not loaded")
        return {"status": "ready"}

    @app.get("/version")
    def version() -> dict:
        return {"app_version": os.getenv("APP_VERSION", "dev"), "model_version": os.getenv("MODEL_VERSION", "unknown"),
                "dataset_version": os.getenv("DATASET_VERSION", "unknown"), "model_loaded": state["model"] is not None}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        model = state["model"]
        if model is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        expected = getattr(model, "n_features_in_", len(req.features))
        if len(req.features) != expected:
            raise HTTPException(status_code=422, detail=f"expected {expected} features, got {len(req.features)}")
        proba = model.predict_proba([req.features])[0]
        return PredictResponse(prediction=int(proba.argmax()), probabilities=[round(float(p), 4) for p in proba],
                               model_version=os.getenv("MODEL_VERSION", "unknown"))

    return app


app = create_app()
