"""
Integration test for FastAPI /spectral and /agriculture endpoints.
"""
from fastapi.testclient import TestClient
from services.models.api.app import app

client = TestClient(app)

def test_spectral_endpoint():
    response = client.post(
        "/spectral",
        data={"query": "Calculate NDVI and vegetation health", "indices_requested": "all"}
    )
    assert response.status_code == 200
    json_data = response.json()
    assert "indices" in json_data
    assert "vegetation_health" in json_data
    assert "reasoning" in json_data
    assert "answer" in json_data

def test_agriculture_endpoint():
    response = client.post(
        "/agriculture",
        data={"query": "Estimate soil moisture, crop yield, and drought risk", "crop_type": "Wheat"}
    )
    assert response.status_code == 200
    json_data = response.json()
    assert "soil_analysis" in json_data
    assert "crop_intelligence" in json_data
    assert "drought_irrigation" in json_data
    assert "answer" in json_data
    assert "Estimate" in json_data["answer"] or "estimate" in json_data["answer"]
