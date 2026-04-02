from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import torch
from model_training import MultimodalHousePredictor

app = FastAPI()

# Load the scaler and model
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

model = MultimodalHousePredictor()
model.load_state_dict(torch.load('house_price_model.pt', map_location=torch.device('cpu')))
model.eval()

class HouseFeatures(BaseModel):
    bhk: int
    bathrooms: float
    sqft: float
    building_type: float
    grade: int
    condition: int
    city_multiplier: float
    lat: float
    lon: float
    sqft_living15: float
    sqft_lot15: float
    city: str

@app.get("/health")
def health_check():
    return {"status": "online", "model_loaded": True}

@app.post("/predict")
def predict(features: HouseFeatures):
    # Your prediction logic here
    # ...
    return {"predicted_price": 5000000.0, "drift_status": "stable"}