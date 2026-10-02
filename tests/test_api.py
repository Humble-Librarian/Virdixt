import pytest
import time
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

def test_analyze_returns_simulator_snapshot():
    """analyze response must include simulator_snapshot with frozen_text_signals."""
    with TestClient(app) as client:
        response = client.post("/api/analyze", data={
            "text": "The company experienced severe liquidity issues.",
            "exposure": 5000000.0
        })
        assert response.status_code == 200
        data = response.json()
        assert "simulator_snapshot" in data, "Missing simulator_snapshot in analyze response"
        snap = data["simulator_snapshot"]
        assert "frozen_text_signals" in snap
        assert "distress_score" in snap["frozen_text_signals"]
        assert "neg_prob" in snap["frozen_text_signals"]
        assert "baseline_rating" in snap
        assert "baseline_priority" in snap

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

def test_simulate_facts_basic():
    """Phase 4: /api/simulate_facts must return verdict + priority_index in < 2s (no ML)."""
    with TestClient(app) as client:
        # First get frozen text signals from analyze
        analyze_resp = client.post("/api/analyze", data={
            "text": "The company experienced severe liquidity issues and revenue decline.",
            "exposure": 5000000.0
        })
        snap = analyze_resp.json()["simulator_snapshot"]
        frozen = snap["frozen_text_signals"]

        financial_facts = {
            "revenue":          50_000_000,
            "net income":       -5_000_000,
            "total assets":    120_000_000,
            "total debt":       80_000_000,
            "cash":              3_000_000,
            "working capital":  -8_000_000,
            "retained earnings": 10_000_000,
        }

        t0 = time.time()
        resp = client.post("/api/simulate_facts", json={
            "financial_facts":     financial_facts,
            "frozen_text_signals": frozen,
            "exposure":            5_000_000.0,
        })
        elapsed = time.time() - t0

        assert resp.status_code == 200, f"simulate_facts failed: {resp.text}"
        data = resp.json()
        assert "verdict" in data
        assert "priority_index" in data
        assert "action_flag" in data
        assert elapsed < 2.0, f"simulate_facts took {elapsed:.2f}s — expected < 2s (no ML)"

def test_simulate_facts_forensic_scores_returned():
    """simulate_facts must return forensic_scores when financial data is provided."""
    with TestClient(app) as client:
        analyze_resp = client.post("/api/analyze", data={
            "text": "Revenue declined and debt covenant was breached.",
            "exposure": 1000000.0
        })
        frozen = analyze_resp.json()["simulator_snapshot"]["frozen_text_signals"]

        resp = client.post("/api/simulate_facts", json={
            "financial_facts": {
                "revenue": 40_000_000, "total assets": 100_000_000,
                "total debt": 60_000_000, "net income": -2_000_000,
                "cash": 5_000_000,
            },
            "frozen_text_signals": frozen,
            "exposure": 1_000_000.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        # Forensic scores should be present since we provided enough financial facts
        assert "forensic_scores" in data

def test_simulate_with_sample_name():
    """Phase 4: simulate endpoint must accept sample_name."""
    with TestClient(app) as client:
        response = client.post("/api/simulate", data={
            "sample_name": "test_scenario.txt",
            "exposure": 5000000.0
        })
        assert response.status_code == 200
        data = response.json()
        assert "priority_index" in data
        assert "verdict" in data
        assert "action_flag" in data

def test_simulate_exposure_changes_priority():
    """Phase 4: higher exposure must produce higher priority_index."""
    with TestClient(app) as client:
        text = "The company experienced severe liquidity issues."
        low  = client.post("/api/simulate", data={"text": text, "exposure": 1000000.0}).json()
        high = client.post("/api/simulate", data={"text": text, "exposure": 50000000.0}).json()
        assert high["priority_index"] > low["priority_index"], \
            "Higher exposure should produce higher priority index"

def test_simulate_speed():
    """Phase 4: after model is warm, simulate must respond in under 10 seconds (ML inference bound)."""
    with TestClient(app) as client:
        # warm-up call
        client.post("/api/simulate", data={
            "text": "Revenue declined significantly this quarter.",
            "exposure": 1000000.0
        })
        # timed call
        t0 = time.time()
        response = client.post("/api/simulate", data={
            "text": "Revenue declined significantly this quarter.",
            "exposure": 2000000.0
        })
        elapsed = time.time() - t0
        assert response.status_code == 200
        assert elapsed < 10.0, f"Simulate took {elapsed:.2f}s — expected < 10s"
