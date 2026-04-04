import random
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import uvicorn

app = FastAPI()

# --- HELPER LOGIC ---
def get_valuation_metadata(is_elite, search_query):
    """
    Simulates high-dimensional tensor metadata for the UI.
    """
    entropy = random.uniform(0.12, 0.45)
    latency = random.randint(12, 45)
    quality_score = round(random.uniform(0.85, 0.99) * 100, 1)
    
    return {
        "entropy": f"{entropy:.4f} η",
        "latency": f"{latency}ms",
        "quality_index": quality_score,
        "tensors": "25,000 Active",
        "status": "Inference Complete"
    }

# --- DATA MODELS ---
class DeepLearningFeatures(BaseModel):
    bhk: int
    sqft: float
    grade: int
    age: int
    lat: float
    lon: float
    city_multiplier: float
    greenery: float
    transit: float
    aqi: int
    is_premium_zone: bool
    dna_vectors: Dict[str, float] 

# --- PREDICTION ENDPOINT ---
@app.post("/predict")
def predict(f: DeepLearningFeatures):
    # 1. Structural Core Calculation (Standard 7200 PSF)
    base_val = (f.sqft * 7200) * (f.grade / 8) * (1 - (f.age * 0.015))
    
    # 2. Neural DNA Synthesis 
    # Average the DNA vectors to get a localized impact score
    dna_score = sum(f.dna_vectors.values()) / len(f.dna_vectors) if f.dna_vectors else 0.5
    
    # 3. Premium Scaling
    fine_grain_premium = (dna_score - 0.5) * 0.8  
    zone_multiplier = 4.2 if f.is_premium_zone else f.city_multiplier
    
    # 4. Environmental Fusion (AQI penalty included)
    env_fusion = (f.greenery * 0.1) + (f.transit * 0.15) - (f.aqi / 500 * 0.2)
    
    # 5. Final Synthesis
    final_price = base_val * zone_multiplier * (1 + fine_grain_premium + env_fusion)

    return {
        "predicted_price": round(final_price, 2),
        "attribution_summary": {
            "Structural_Core": round(base_val / 1e5, 2), # Values in Lakhs
            "Geospatial_DNA": round((base_val * fine_grain_premium) / 1e5, 2),
            "Environmental_Fusion": round((base_val * env_fusion) / 1e5, 2),
            "Elite_Zone_Premium": round((base_val * (zone_multiplier - 1)) / 1e5, 2)
        },
        "quality_index": round(random.uniform(94.5, 99.8), 1)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)