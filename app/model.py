from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


class FraudPredictor:
    def __init__(self, artifact_path: str | None = None) -> None:
        self.artifact_path = artifact_path or os.getenv(
            "MODEL_PATH", "artifacts/fraud_model.joblib"
        )
        self.artifact: dict[str, Any] | None = None
        self.reload()

    def reload(self) -> None:
        path = Path(self.artifact_path)
        self.artifact = joblib.load(path) if path.exists() else None

    @property
    def ready(self) -> bool:
        return self.artifact is not None

    def info(self) -> dict[str, Any]:
        if not self.artifact:
            return {"ready": False, "artifact": self.artifact_path}
        return {
            "ready": True,
            "artifact": self.artifact_path,
            "feature_count": len(self.artifact["features"]),
            "threshold": self.artifact["threshold"],
            "metrics": self.artifact.get("metrics", {}),
        }

    def predict(self, features: dict[str, float]) -> dict[str, Any]:
        if not self.artifact:
            raise RuntimeError(
                "Model artifact not found. Run train.py before serving predictions."
            )

        expected = self.artifact["features"]
        missing = [name for name in expected if name not in features]
        extra = [name for name in features if name not in expected]
        if missing or extra:
            raise ValueError({"missing_features": missing, "unexpected_features": extra})

        frame = pd.DataFrame([[features[name] for name in expected]], columns=expected)
        probability = float(self.artifact["model"].predict_proba(frame)[0, 1])
        threshold = float(self.artifact["threshold"])
        return {
            "fraud_probability": round(probability, 6),
            "is_fraud": probability >= threshold,
            "threshold": threshold,
        }
