from fastapi.testclient import TestClient

from app.main import app, predictor

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info() -> None:
    response = client.get("/model-info")
    assert response.status_code == 200
    assert "ready" in response.json()


def test_predict_without_artifact_returns_503(monkeypatch) -> None:
    monkeypatch.setattr(predictor, "artifact", None)
    response = client.post("/predict", json={"features": {"Time": 1.0}})
    assert response.status_code == 503
