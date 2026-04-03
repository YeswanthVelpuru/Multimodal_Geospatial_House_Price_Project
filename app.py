import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.graph_objects as go
import requests
import os
import random
from datetime import datetime
from geopy.geocoders import Nominatim

st.set_page_config(page_title="Urban-AI Deep Intelligence", layout="wide")

# Mapbox Configuration
DEFAULT_TOKEN = "pk.eyJ1IjoieWVzd2FudGgtLXYtLTIwMDMiLCJhIjoiY21taHh5ZmJtMHRneDJwczZxaWhiYmg3ZiJ9.IZK_WUOAlFdAsg0ewYyARg"
MAPBOX_TOKEN = DEFAULT_TOKEN
if os.path.exists(".streamlit/secrets.toml"):
    try: MAPBOX_TOKEN = st.secrets["MAPBOX_TOKEN"]
    except: pass

# --- JEWEL TONE PALETTE ---
# Amethyst (#9966cc), Sapphire (#0f0c29), Emerald (#50C878)
st.markdown("""
    <style>
    .stApp { background: radial-gradient(circle at top right, #0d0d2b, #050514); color: #e0e0e0; }
    .glass-card { 
        background: rgba(153, 102, 204, 0.03); 
        border: 1px solid rgba(80, 200, 120, 0.25); 
        padding: 30px; border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.6);
    }
    .metric-val { 
        font-size: 3.5rem; color: #50C878; font-weight: bold; 
        text-shadow: 0 0 25px rgba(80, 200, 120, 0.5); 
    }
    .section-header { color: #9966cc; font-weight: bold; letter-spacing: 1px; margin-bottom: 15px; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h1 style='color:#50C878;'>🛰️ SYSTEM CONTROL</h1>", unsafe_allow_html=True)
    search_query = st.text_input("📍 LOCATION SEARCH", "Nellore, Andhra Pradesh")
    
    st.markdown("<div class='section-header'>🏗️ STRUCTURAL TENSORS</div>", unsafe_allow_html=True)
    bhk = st.slider("BHK", 1, 10, 3)
    sqft = st.number_input("Square Footage", 500, 15000, 2400)
    grade = st.slider("Construction Grade", 1, 13, 9)
    age = st.slider("Property Age", 0, 50, 5)

    st.markdown("<div class='section-header'>🍃 ECO & SOCIAL DNA</div>", unsafe_allow_html=True)
    greenery = st.slider("Greenery (NDVI)", 0.0, 1.0, 0.6)
    water = st.slider("Water Security", 0.0, 1.0, 0.8)
    transit = st.slider("Transit Access", 0.0, 1.0, 0.7)
    aqi = st.slider("AQI Level", 20, 500, 55)
    flood = st.slider("Flood Risk", 0.0, 1.0, 0.1)

# --- BACKGROUND DNA GENERATION (Hidden from UI) ---
random.seed(len(search_query))
dna_vectors = {f"F{i}": random.uniform(0.3, 0.9) for i in range(60)}

# --- GEOLOCATION ---
geolocator = Nominatim(user_agent="urban_clean_75")
try:
    loc = geolocator.geocode(search_query)
    lat, lon = (loc.latitude, loc.longitude) if loc else (14.4426, 79.9865)
except: lat, lon = 14.4426, 79.9865

# --- DASHBOARD ---
st.markdown("<h1 style='text-align:center; color:#9966cc; font-family:serif;'>MULTIMODAL GEOSPATIAL INTELLIGENCE</h1>", unsafe_allow_html=True)
st.markdown("---")

col_main, col_stats = st.columns([1.8, 1])

with col_main:
    # 1. Satellite Map
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/satellite-streets-v12',
        api_keys={'mapbox': MAPBOX_TOKEN},
        initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=16, pitch=45),
        layers=[pdk.Layer("ScatterplotLayer", data=pd.DataFrame({'lat':[lat], 'lon':[lon]}), 
                get_position='[lon, lat]', get_color='[80, 200, 120, 200]', get_radius=40)]
    ))
    
    # 2. Quality Radar
    st.markdown("<div class='section-header'>🌀 URBAN QUALITY PULSE</div>", unsafe_allow_html=True)
    radar_fig = go.Figure(data=go.Scatterpolar(
        r=[greenery, transit, (500-aqi)/500, 0.8, (1-flood), water],
        theta=['Eco', 'Transit', 'Air Quality', 'Stability', 'Safety', 'Water'],
        fill='toself', fillcolor='rgba(153, 102, 204, 0.2)', line_color='#9966cc'
    ))
    radar_fig.update_layout(polar=dict(radialaxis=dict(visible=False)), paper_bgcolor='rgba(0,0,0,0)', 
                            font_color="white", height=350, margin=dict(l=50,r=50,t=20,b=20))
    st.plotly_chart(radar_fig, use_container_width=True)

with col_stats:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<p style='color:#50C878; font-weight:bold; letter-spacing:2px;'>VALUATION CORE</p>", unsafe_allow_html=True)
    
    if st.button("EXECUTE DEEP INFERENCE"):
        payload = {
            "bhk": bhk, "sqft": float(sqft), "grade": grade, "age": age,
            "lat": lat, "lon": lon, "city_multiplier": 1.25,
            "greenery": greenery, "water": water, "transit": transit,
            "aqi": aqi, "flood_risk": flood,
            "hospitals": 3, "education": 5, "comm_density": 0.5, # Fixed defaults for hidden logic
            "dna_vectors": dna_vectors
        }
        
        try:
            res = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=10)
            if res.status_code == 200:
                data = res.json()
                st.markdown(f"<p class='metric-val'>₹{data['predicted_price']:,}</p>", unsafe_allow_html=True)
                st.write(f"🏙️ **Urban DNA Integrity:** {data['quality_index']}%")
                
                # Waterfall Chart
                contribs = data['contributions']
                fig = go.Figure(go.Waterfall(
                    orientation="h", y=list(contribs.keys()), x=list(contribs.values()),
                    connector={"line":{"color":"#9966cc"}},
                    increasing={"marker":{"color":"#50C878"}}, decreasing={"marker":{"color":"#FF4B4B"}}
                ))
                fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', font_color="white", margin=dict(l=0,r=0,t=30,b=0))
                st.plotly_chart(fig, use_container_width=True)
            else: st.error("Inference Error")
        except: st.error("🔌 API Offline")
    
    st.markdown("---")
    st.write("🤖 **Neural Engine:** v4.2 Deep-Geospatial")
    st.write("💎 **Aesthetic:** Jewel-Tone Multimodal")
    st.markdown('</div>', unsafe_allow_html=True)