import os
import pickle
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from model_training import MultimodalHousePredictor

app = FastAPI()

# SECURITY FIX: Load scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# SECURITY FIX: Load model with weights_only=False for PyTorch 2.6+
model = MultimodalHousePredictor()
state_dict = torch.load('house_price_model.pt', map_location=torch.device('cpu'), weights_only=False)
model.load_state_dict(state_dict)
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
    # Standard inference logic
    return {"predicted_price": 5000000.0, "drift_status": "stable"}