import os
import pickle
import torch
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict

app = FastAPI()

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None
    }

# Load Scaler/Model
scaler = None
if os.path.exists('scaler.pkl'):
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

model = None
if os.path.exists('house_price_model.pt'):
    try: model = torch.jit.load('house_price_model.pt', map_location=torch.device('cpu'))
    except: model = None

class MassiveFeatureSet(BaseModel):
    bhk: int; sqft: float; grade: int; age: int
    lat: float; lon: float; city_multiplier: float
    greenery: float; water: float; transit: float
    aqi: int; flood_risk: float; hospitals: int; education: int; comm_density: float
    dna_vectors: Dict[str, float] # The 60 hidden features

@app.post("/predict")
def predict(f: MassiveFeatureSet):
    # 1. Base Structural Value
    base_val = (f.sqft * 4800) * (f.grade / 8) * (1 - (f.age * 0.01)) * f.city_multiplier
    
    # 2. Process Hidden DNA (60 features) silently
    dna_score = sum(f.dna_vectors.values()) / 60
    dna_premium = (dna_score - 0.5) * 0.35 
    
    # 3. Process Visible Features
    env_impact = (f.greenery * 0.06) - (f.aqi / 500 * 0.1) - (f.flood_risk * 0.15)
    soc_impact = (f.transit * 0.08) + (f.hospitals * 0.02) + (f.education * 0.03)
    
    final_price = base_val * (1 + dna_premium + env_impact + soc_impact)

    return {
        "predicted_price": round(final_price, 2),
        "contributions": {
            "Core Structure": base_val,
            "Urban Infrastructure (DNA)": base_val * (dna_premium + soc_impact),
            "Environmental DNA": base_val * env_impact
        },
        "quality_index": round(dna_score * 100, 1)
    }

    @app.get("/")
def read_root():
    return {"message": "Multimodal Geospatial API is running"}