import streamlit as st
import torch
import pickle
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

from model_training import MultimodalHousePredictor
from geopy.geocoders import Nominatim

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Urban Intelligence Dashboard", layout="wide")
geolocator = Nominatim(user_agent="urban_ai_final_v3")

STRUCTURAL_FEATURES = ['BHK', 'Bathrooms', 'Living Area', 'Building Type', 'Grade', 'Condition', 'City Multiplier']
GEOSPATIAL_FEATURES = ['Latitude', 'Longitude', 'Living Area 2015', 'Lot Area 2015']
ALL_FEATURES = STRUCTURAL_FEATURES + GEOSPATIAL_FEATURES

BUILDING_TYPES = {
    "Apartment": 1.0,
    "High Rise Apartment": 1.25,
    "Independent House": 1.15,
    "Villa": 1.9
}

# -----------------------------
# LOAD MODEL (FOR SHAP ONLY)
# -----------------------------
@st.cache_resource
def load_assets():
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    model = MultimodalHousePredictor(struct_dim=7, geo_dim=4)
    model.load_state_dict(torch.load('house_price_model.pth'))
    model.eval()

    return scaler, model

scaler, model = load_assets()

# -----------------------------
# UI
# -----------------------------
st.title("🏠 Multimodal Geospatial Deep Learning for Urban House Price Prediction")

col_input, col_viz = st.columns([1, 1.3])

# -----------------------------
# INPUT SECTION
# -----------------------------
with col_input:
    locality = st.text_input("Locality", "Kelambakkam")
    selected_city = st.selectbox("City", ["Chennai", "Hyderabad", "Delhi"])

    t_mul = 1.6 if selected_city == "Chennai" else 1.0

    bhk = st.number_input("BHK", 1, 10, 3)
    b_type = st.selectbox("Building Type", list(BUILDING_TYPES.keys()))
    sqft = st.number_input("Built-up Area (Sqft)", 500, 15000, 2000)

    if st.button("Generate AI Valuation"):

        full_address = f"{locality}, {selected_city}"
        location = geolocator.geocode(full_address)
        lat, lon = (location.latitude, location.longitude) if location else (13.08, 80.27)

        # -----------------------------
        # REST API CALL
        # -----------------------------
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={
                "bhk": bhk,
                "bathrooms": max(1, bhk - 0.5),
                "sqft": sqft,
                "building_type": BUILDING_TYPES[b_type],
                "grade": 7,
                "condition": 3,
                "city_multiplier": t_mul,
                "lat": lat,
                "lon": lon,
                "sqft_living15": sqft,
                "sqft_lot15": 5000,
                "city": selected_city
            }
        )

        if response.status_code != 200:
            st.error("❌ API not running. Start FastAPI first.")
            st.stop()

        data = response.json()

        st.session_state.results = {
            "price": data["predicted_price"],
            "rate": data["price_per_sqft"],
            "drift": data["drift_status"],
            "lat": lat,
            "lon": lon,
            "address": full_address,
            "inputs": {
                "bhk": bhk,
                "sqft": sqft,
                "b_type": b_type,
                "t_mul": t_mul
            }
        }

# -----------------------------
# OUTPUT SECTION
# -----------------------------
with col_viz:
    if "results" in st.session_state:
        res = st.session_state.results

        st.subheader(f"📍 {res['address']}")

        st.metric("Estimated Price", f"₹{res['price']:,.0f}")
        st.metric("Price per Sqft", f"₹{res['rate']:,.2f}")
        st.metric("Drift Status", res["drift"])

        # Map
        st.map(pd.DataFrame({'lat': [res['lat']], 'lon': [res['lon']]}))

        # -----------------------------
        # SHAP ANALYSIS
        # -----------------------------
        st.divider()
        st.subheader("🔍 Features Affecting Price")

        def model_predict(data):
            with torch.no_grad():
                return model(
                    torch.tensor(data[:, :7], dtype=torch.float32),
                    torch.tensor(data[:, 7:], dtype=torch.float32)
                ).detach().numpy()

        inp = res["inputs"]

        raw_x = np.array([[
            inp["bhk"],
            max(1, inp["bhk"] - 0.5),
            inp["sqft"],
            BUILDING_TYPES[inp["b_type"]],
            7,
            3,
            inp["t_mul"],
            res["lat"],
            res["lon"],
            inp["sqft"],
            5000
        ]])

        scaled_x = scaler.transform(raw_x)

        background = scaled_x + np.random.normal(0, 0.05, (10, 11))
        explainer = shap.KernelExplainer(model_predict, background)
        shap_values = explainer.shap_values(scaled_x)

        vals = shap_values[0].flatten() if isinstance(shap_values, list) else shap_values.flatten()

        importance_df = pd.DataFrame({
            "Feature": ALL_FEATURES,
            "Impact": vals
        })

        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        importance_df.sort_values(by="Impact").plot.barh(
            x="Feature", y="Impact", ax=ax, legend=False
        )
        ax.set_title("Feature Contribution to Price")
        st.pyplot(fig)

        # -----------------------------
        # TOP 5 % IMPACT
        # -----------------------------
        st.subheader("📊 Top 5 Price Influencing Factors")

        abs_vals = np.abs(vals)
        total = np.sum(abs_vals)

        importance_df["Percent"] = (abs_vals / total) * 100
        importance_df["AbsImpact"] = abs_vals

        top5 = importance_df.sort_values(by="AbsImpact", ascending=False).head(5)

        for _, row in top5.iterrows():
            direction = "⬆️" if row["Impact"] > 0 else "⬇️"
            st.write(f"{direction} **{row['Feature']}** → {row['Percent']:.2f}%")

        # -----------------------------
        # CLEAN INSIGHTS
        # -----------------------------
        st.subheader("🧠 Key Insights")

        top_positive = importance_df.sort_values(by="Impact", ascending=False).head(3)["Feature"].values
        top_negative = importance_df.sort_values(by="Impact").head(3)["Feature"].values

        st.write("⬆️ **Top Positive Drivers:**", ", ".join(top_positive))
        st.write("⬇️ **Top Negative Drivers:**", ", ".join(top_negative))
