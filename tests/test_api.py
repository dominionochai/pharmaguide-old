from fastapi.testclient import TestClient

from api import main


client = TestClient(main.app)


def test_health_is_available_without_a_trained_model():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert isinstance(response.json()["model_loaded"], bool)
    assert response.json()["medicine_count"] > 0


def test_detect_requires_two_medicines():
    response = client.post("/detect", json={"medications": ["ASPIRIN"]})

    assert response.status_code == 422


def test_detect_is_case_insensitive_and_returns_database_flags(monkeypatch):
    monkeypatch.setattr(main, "_load_model", lambda: None)

    response = client.post(
        "/detect",
        json={"medications": ["AsPiRiN", "wArFaRiN"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "database_only"
    assert body["model_loaded"] is False
    assert [item["name"] for item in body["matched"]] == ["aspirin", "warfarin"]
    assert body["interactions"]
    flag = body["interactions"][0]
    assert flag["drug_a"] == "aspirin"
    assert flag["drug_b"] == "warfarin"
    assert flag["severity"] == "major"
    assert flag["confidence"] == 0.95
    assert flag["reason"]
    assert flag["source"] == "medicine_database"


def test_detect_reports_unknown_items():
    response = client.post(
        "/detect",
        json={"medications": ["aspirin", "not-a-real-medicine"]},
    )

    assert response.status_code == 400
    assert "not-a-real-medicine" in response.json()["detail"]


def test_herbs_endpoint_returns_the_local_catalogue():
    response = client.get("/herbs")

    assert response.status_code == 200
    assert isinstance(response.json(), list)
