from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_read_health():
    # This will now pass because we added the /health route
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_prediction_endpoint():
    # Test a sample prediction payload
    payload = {
        "bhk": 3, "bathrooms": 2.5, "sqft": 2400.0, "building_type": 1.0,
        "grade": 8, "condition": 4, "city_multiplier": 1.2,
        "lat": 14.44, "lon": 79.98, "sqft_living15": 2100.0,
        "sqft_lot15": 4500.0, "city": "Nellore"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predicted_price" in response.json()