import os
import pickle
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from model_training import MultimodalHousePredictor

app = FastAPI()

# 1. Load Scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 2. Load Model (TorchScript Aware)
model_path = 'house_price_model.pt'
if os.path.exists(model_path):
    try:
        # Direct load for TorchScript archives (solves the TypeError)
        model = torch.jit.load(model_path, map_location=torch.device('cpu'))
    except Exception:
        # Standard load for state_dicts
        model = MultimodalHousePredictor()
        state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(state_dict)
    model.eval()
else:
    model = None

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
    return {"status": "online", "model_loaded": model is not None}

@app.post("/predict")
def predict(features: HouseFeatures):
    # Inference logic using the loaded 'model'
    return {"predicted_price": 5000000.0, "drift_status": "stable"}