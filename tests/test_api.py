from fastapi.testclient import TestClient

from api import main


client = TestClient(main.app)


def test_health_is_available_without_a_trained_model():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert isinstance(response.json()["model_loaded"], bool)


def test_detect_requires_two_medications():
    response = client.post("/detect", json={"medications": ["aspirin"]})

    assert response.status_code == 422


def test_detect_reports_missing_model(monkeypatch):
    monkeypatch.setattr(main, "_load_model", lambda: None)

    response = client.post(
        "/detect", json={"medications": ["aspirin", "warfarin"]}
    )

    assert response.status_code == 400
    assert "not trained" in response.json()["detail"]


def test_herbs_endpoint_returns_the_local_catalogue():
    response = client.get("/herbs")

    assert response.status_code == 200
    assert isinstance(response.json(), list)
