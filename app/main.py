from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.model import FraudPredictor

app = FastAPI(
    title="Real-Time Fraud Detection Microservice",
    version="1.0.0",
    description="Score credit-card transactions using a trained fraud model artifact.",
)
predictor = FraudPredictor()


class TransactionRequest(BaseModel):
    features: dict[str, float] = Field(min_length=1)


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {"status": "ok", "model_ready": predictor.ready}


@app.get("/model-info")
def model_info() -> dict:
    return predictor.info()


@app.post("/predict")
def predict(payload: TransactionRequest) -> dict:
    if not predictor.ready:
        raise HTTPException(
            status_code=503,
            detail="Model artifact not found. Run train.py and restart the service.",
        )
    try:
        return predictor.predict(payload.features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=exc.args[0]) from exc
