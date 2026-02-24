import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "ok"

def test_process_text():
    payload = {
        "message_id": "test-123",
        "user_id": "user-456",
        "text": "Me siento muy cansado y tengo dolor de espalda",
        "timestamp": "2026-02-06T10:00:00Z",
        "language_hint": "es"
    }
    response = client.post("/v1/process", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # Contract verification
    assert data["message_id"] == "test-123"
    assert "embeddings" in data
    assert data["embeddings"]["dim"] == 384
    assert len(data["embeddings"]["vector"]) == 384
    
    assert "symptom_scores" in data
    for symptom in ["pain", "fatigue", "anxiety", "mood", "sleep"]:
        assert symptom in data["symptom_scores"]
        score = data["symptom_scores"][symptom]
        assert 0 <= score["prob"] <= 1
        assert "uncertainty" in score

def test_empty_text():
    payload = {
        "message_id": "test-empty",
        "user_id": "user-456",
        "text": "",
        "language_hint": "es"
    }
    response = client.post("/v1/process", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert all(v == 0.0 for v in data["embeddings"]["vector"])
