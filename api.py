from fastapi import FastAPI
from pydantic import BaseModel
import torch
import pickle
import numpy as np

from model_training import MultimodalHousePredictor
from rl_price_trend import RLPriceAgent
from market_features import scrape_market_trends

app = FastAPI(title="House Price Prediction API")

# -----------------------------
# Load Assets (same as Streamlit)
# -----------------------------
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

model = torch.jit.load("house_price_model.pt")
model.eval()

monitor = RLPriceAgent()

# -----------------------------
# Request Schema
# -----------------------------
class HouseInput(BaseModel):
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


# -----------------------------
# Root Health Check
# -----------------------------
@app.get("/")
def home():
    return {"status": "API is running"}


# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict(data: HouseInput):
    
    # -----------------------------
    # Prepare Input
    # -----------------------------
    raw_x = np.array([[ 
        data.bhk,
        data.bathrooms,
        data.sqft,
        data.building_type,
        data.grade,
        data.condition,
        data.city_multiplier,
        data.lat,
        data.lon,
        data.sqft_living15,
        data.sqft_lot15
    ]])

    scaled_x = scaler.transform(raw_x)

    # -----------------------------
    # Model Inference
    # -----------------------------
    with torch.no_grad():
        s_t = torch.tensor(scaled_x[:, :7], dtype=torch.float32)
        g_t = torch.tensor(scaled_x[:, 7:], dtype=torch.float32)
        price = model(s_t, g_t).item()

    # -----------------------------
    # 🔥 Post-Model Price Adjustment (YOUR ADDITION)
    # -----------------------------
    price = price * data.city_multiplier * data.building_type

    # -----------------------------
    # Monitoring Logic
    # -----------------------------
    unit_rate = price / data.sqft if data.sqft > 0 else 0
    live_rate = scrape_market_trends(data.city)
    drift_status = monitor.monitor_drift(unit_rate, live_rate)

    # -----------------------------
    # Response
    # -----------------------------
    return {
        "predicted_price": round(price, 2),
        "price_per_sqft": round(unit_rate, 2),
        "market_rate": live_rate,
        "drift_status": drift_status
    }
