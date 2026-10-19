"""HTTP API for the SMS classifier.

POST /predict  {"text": "..."} -> {"label": "ham|spam|smishing", "scores": {...}}
GET  /healthz  ready once the model is loaded
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from app.inference import Classifier


class Predicts(Protocol):
    def predict(self, texts: list[str]) -> list[tuple[str, dict[str, float]]]: ...


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


class PredictResponse(BaseModel):
    label: str
    scores: dict[str, float]


def create_app(classifier: Predicts | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.classifier = classifier or Classifier(Path(os.environ.get("MODEL_DIR", "/model")))
        yield

    app = FastAPI(title="SMS classifier", lifespan=lifespan)

    # A plain def runs in FastAPI's thread pool, which suits CPU-bound ONNX inference.
    @app.post("/predict", response_model=PredictResponse)
    def predict(body: PredictRequest, request: Request) -> PredictResponse:
        label, scores = request.app.state.classifier.predict([body.text])[0]
        return PredictResponse(label=label, scores=scores)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
