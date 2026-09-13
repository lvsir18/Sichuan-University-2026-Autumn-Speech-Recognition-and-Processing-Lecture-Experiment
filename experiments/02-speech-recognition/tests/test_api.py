from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_demo_recognition_is_labeled_simulated():
    response = client.post("/api/recognize/demo", json={"sample_id": "campus", "model": "tiny", "beam_size": 1})
    assert response.status_code == 200
    assert response.json()["metadata"]["mode"] == "simulated"


def test_decode_validation():
    response = client.post("/api/decode", json={"beam_width": 0, "lm_weight": 0})
    assert response.status_code == 422


def test_dataset_status_before_download(tmp_path, monkeypatch):
    monkeypatch.setattr("app.main.DATASET_OUTPUT", tmp_path / "not-prepared")
    response = client.get("/api/dataset/status")
    assert response.status_code == 200
    assert response.json()["ready"] is False


def test_dataset_recognition_requires_prepared_data(tmp_path, monkeypatch):
    monkeypatch.setattr("app.main.DATASET_OUTPUT", tmp_path / "not-prepared")
    response = client.post("/api/recognize/dataset", json={"sample_id": "fleurs-0000", "model": "tiny", "beam_size": 1})
    assert response.status_code == 409
