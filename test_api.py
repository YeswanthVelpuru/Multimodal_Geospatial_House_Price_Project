from fastapi.testclient import TestClient
from api import app
from datetime import datetime

client = TestClient(app)

def test_read_health():
    # Attempt to hit the health endpoint
    response = client.get("/health")
    
    # If /health is missing, try the root /
    if response.status_code == 404:
        response = client.get("/")
        
    # If it's still 404, the API isn't exposing routes correctly
    assert response.status_code == 200, f"Expected 200 but got {response.status_code}. Content: {response.text}"

def test_prediction_endpoint():
    # Generate 60 hidden features
    dna_vectors = {f"F{i}": 0.5 for i in range(60)}
    
    payload = {
        "bhk": 3, 
        "sqft": 2400.0, 
        "grade": 9, 
        "age": 5,
        "lat": 14.44, 
        "lon": 79.98, 
        "city_multiplier": 1.25,
        "greenery": 0.6, 
        "water": 0.8, 
        "transit": 0.7,
        "aqi": 55, 
        "flood_risk": 0.1, 
        "hospitals": 3, 
        "education": 5, 
        "comm_density": 0.5,
        "dna_vectors": dna_vectors
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price" in data