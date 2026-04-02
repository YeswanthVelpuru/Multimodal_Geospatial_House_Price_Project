from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_read_health():
    # Tests if the health endpoint we created works
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "Online" or response.json()["status"] == "Degraded (No Model)"