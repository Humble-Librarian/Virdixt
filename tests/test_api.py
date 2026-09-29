import pytest
from fastapi.testclient import TestClient
from server import app

def test_list_samples():
    with TestClient(app) as client:
        response = client.get("/api/samples")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        if len(data) > 0:
            assert "filename" in data[0]
            assert "size" in data[0]

def test_analyze_text():
    with TestClient(app) as client:
        response = client.post("/api/analyze", data={
            "text": "The company experienced severe liquidity issues.",
            "exposure": 5000000.0
        })
        assert response.status_code == 200
        data = response.json()
        assert "verdict" in data
        assert "trace_logs" in data
        assert "priority_index" in data
        assert data["priority_index"] > 0

def test_simulate():
    with TestClient(app) as client:
        response = client.post("/api/simulate", data={
            "text": "The company experienced severe liquidity issues.",
            "exposure": 10000000.0
        })
        assert response.status_code == 200
        data = response.json()
        assert "priority_index" in data
        assert "verdict" in data
