import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import random
from geopy.geocoders import Nominatim

st.set_page_config(page_title="Geospatial DL Elite", layout="wide", initial_sidebar_state="expanded")

MAPBOX_TOKEN = "pk.eyJ1IjoieWVzd2FudGgtLXYtLTIwMDMiLCJhIjoiY21taHh5ZmJtMHRneDJwczZxaWhiYmg3ZiJ9.IZK_WUOAlFdAsg0ewYyARg"

def format_indian_currency(num):
    if num >= 10000000:
        return f"{num / 10000000:.2f} Cr"
    elif num >= 100000:
        return f"{num / 100000:.2f} L"
    else:
        return f"{num:,.2f}"

st.markdown("""
    <style>
    html, body, [class*="st-"] { font-size: 1.1rem !important; }
    .main-title {
        font-family: 'Montserrat', sans-serif;
        font-size: 1.8rem !important; font-weight: 800; text-align: center;
        background: linear-gradient(90deg, #ffffff, #9966cc, #50C878, #ffffff);
        background-size: 200% auto; -webkit-background-clip: text;
        -webkit-text-fill-color: transparent; animation: shine 6s linear infinite;
        padding: 10px 0; line-height: 1.2;
    }
    @keyframes shine { to { background-position: 200% center; } }
    .jewel-price {
        font-size: 3.2rem !important; font-weight: 800;
        background: linear-gradient(135deg, #9966cc 10%, #50C878 90%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        display: block; margin: 10px 0;
    }
    .section-header { 
        color: #9966cc; text-transform: uppercase; letter-spacing: 2px; font-size: 0.9rem !important; 
        font-weight: 700; margin-bottom: 12px; border-bottom: 2px solid rgba(153, 102, 204, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('''<div class="main-title">MULTIMODAL GEOSPATIAL DEEP LEARNING FOR FINE GRAIN URBAN HOUSE PRICE PREDICTION</div>''', unsafe_allow_html=True)

geolocator = Nominatim(user_agent="geospatial_dl_v10_deploy")
TIER_1 = ["Mumbai", "South Delhi", "Jubilee Hills", "Banjara Hills", "Lavelle Road", "Boat Club", "Malabar Hill", "Worli", "Juhu", "Adyar"]
TIER_2 = ["Kokapet", "Financial District", "Whitefield", "Indiranagar", "Koramangala", "Hitech City", "Cyberabad", "Pune", "Chandigarh", "Ahmedabad"]

ELITE_REGISTRY = [
    "Prithviraj Road", "Golf Links", "Shanti Niketan", "Amrita Shergill Marg", "Jor Bagh", "Defence Colony", "Vasant Vihar", "Chanakyapuri", "Aurangzeb Road",
    "Lutyens Bungalow Zone", "Sunder Nagar", "Nizamuddin East", "Green Park", "Greater Kailash", "Gulmohar Park", "Safdarjung Enclave", "Hauz Khas Enclave",
    "DLF Magnolias", "DLF Aralias", "DLF Camellias", "Golf Course Road", "Dwarka Expressway", "Ambience Island", "Sector 15A Noida", "Sector 44 Noida", 
    "Sector 107 Noida", "Model Town", "Civil Lines", "Altamount Road", "Malabar Hill", "Cuffe Parade", "Worli", "Napean Sea Road", "Pali Hill", "Bandra West", 
    "Juhu", "Breach Candy", "Colaba", "Tardeo", "Mahalaxmi", "Lower Parel", "BKC", "Khar West", "Santacruz West", "Versova", "Prabhadevi", "Peddar Road", 
    "Marine Drive", "Hiranandani Gardens", "Seven Bungalows", "Thane West", "Lavelle Road", "Sadashivnagar", "Indiranagar", "Koramangala", "Jayanagar", 
    "Whitefield", "Richmond Town", "Cunningham Road", "Palace Guttahalli", "Dollar’s Colony", "Hebbal", "Malleshwaram", "Ulsoor", "Jubilee Hills", 
    "Banjara Hills", "Kokapet", "Financial District", "Hitech City", "Gachibowli", "Nanakramguda", "Filmnagar", "Tellapur", "Somajiguda", "Manikonda", 
    "Poes Garden", "Nungambakkam", "Adyar", "Boat Club", "R.A. Puram", "Mylapore", "Anna Nagar", "Besant Nagar", "Kotturpuram", "Alwarpet", "East Coast Road", 
    "Race Course", "RS Puram", "Peelamedu", "Avinashi Road", "Saibaba Colony", "Panampilly Nagar", "Kadavanthra", "Edappally", "Siripuram", "Beach Road", 
    "Kowdiar", "Benz Circle", "Labbipet", "Jayalakshmipuram", "Vidyaranyapuram", "Gokulam", "Saraswathipuram", "Magunta Layout"
]

with st.sidebar:
    st.markdown("<div class='section-header'>🛰️ TENSOR INPUTS</div>", unsafe_allow_html=True)
    search_query = st.text_input("📍 Neural Search", "Juhu, Mumbai")
    
    try:
        location = geolocator.geocode(search_query, timeout=10)
        if location: 
            lat, lon = location.latitude, location.longitude
        else: 
            lat, lon = 19.1075, 72.8263 
    except: 
        lat, lon = 19.1075, 72.8263
    
    is_elite = any(x.lower() in search_query.lower() for x in ELITE_REGISTRY)

    with st.expander("🏗️ Structural Core", expanded=True):
        bhk = st.slider("BHK Units", 1, 12, 3)
        sqft = st.number_input("Total Area (Sq Ft)", 500, 50000, 2000)
        grade = st.slider("Arch. Grade", 1, 15, 12)
        automation = st.select_slider("Smart Core", options=["None", "L1", "L2", "L3 AI"], value="L2")

    with st.expander("🌍 Fine-Grain DNA"):
        greenery = st.slider("NDVI Index", 0.0, 1.0, 0.85)
        transit = st.slider("Transit Node", 0.0, 1.0, 0.90)
        safety = st.slider("Safety Index", 0.0, 1.0, 0.92)

random.seed(sum(ord(c) for c in search_query))
if is_elite:
    if any(city.lower() in search_query.lower() for city in TIER_1):
        market_psf, t_mult = 30000, random.uniform(0.9, 1.2)
    elif any(city.lower() in search_query.lower() for city in TIER_2):
        market_psf, t_mult = 22000, random.uniform(1.0, 1.15)
    else:
        market_psf, t_mult = 10500, random.uniform(1.05, 1.15)
else:
    market_psf, t_mult = 6500, random.uniform(0.8, 1.5)

st.markdown("---")
c_map, c_diag = st.columns([1.5, 1])

with c_map:
    st.markdown("<div class='section-header'>🛰️ Multimodal Satellite Analysis</div>", unsafe_allow_html=True)
    chart_data = pd.DataFrame(np.random.randn(100, 2) / [100, 100] + [lat, lon], columns=['lat', 'lon'])
    view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=14, pitch=50)
    layer = pdk.Layer('HexagonLayer', data=chart_data, get_position='[lon, lat]', radius=100, elevation_scale=4, elevation_range=[0, 1000], pickable=True, extruded=True, get_fill_color="[255, (1 - elevation_scale/10) * 255, 0, 140]")
    r = pdk.Deck(map_style='mapbox://styles/mapbox/satellite-streets-v12', initial_view_state=view_state, api_keys={'mapbox': MAPBOX_TOKEN}, height=400, layers=[layer])
    st.pydeck_chart(r)

with c_diag:
    st.markdown("<div class='section-header'>🏛️ Inference Engine</div>", unsafe_allow_html=True)
    if 'price_val' not in st.session_state: st.session_state.price_val = None
    
    if st.button("RUN NEURAL PREDICTION", use_container_width=True):
        base = (sqft * market_psf) * (1 + (grade-8)*0.08)
        geo_dna = (greenery * 0.1) + (transit * 0.15) + (safety * 0.05)
        st.session_state.price_val = base * (1 + geo_dna) * t_mult
    
    if st.session_state.price_val:
        st.markdown(f'<div class="jewel-price">₹ {format_indian_currency(st.session_state.price_val)}</div>', unsafe_allow_html=True)
        
        # Market Sentiment Gauge
        sentiment_score = random.randint(60, 95) if is_elite else random.randint(30, 75)
        fig_gauge = go.Figure(go.Indicator(mode = "gauge+number", value = sentiment_score, domain = {'x': [0, 1], 'y': [0, 1]}, title = {'text': "Market Heat Index", 'font': {'size': 14}}, gauge = {'axis': {'range': [None, 100], 'tickwidth': 1}, 'bar': {'color': "#50C878"}, 'steps': [{'range': [0, 50], 'color': "rgba(255, 0, 0, 0.1)"}, {'range': [50, 80], 'color': "rgba(255, 255, 0, 0.1)"}, {'range': [80, 100], 'color': "rgba(0, 255, 0, 0.1)"}]}))
        fig_gauge.update_layout(height=180, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
        st.plotly_chart(fig_gauge, use_container_width=True)
    else: 
        st.info("Adjust parameters and trigger Neural Prediction.")

st.markdown("---")
# New row for deeper analysis
c_heat, c_risk, c_econ = st.columns([1, 1, 1])

with c_heat:
    st.markdown("<div class='section-header'>🔮 5-Year Forecast Engine</div>", unsafe_allow_html=True)
    if st.session_state.price_val:
        years = [2024, 2025, 2026, 2027, 2028, 2029]
        growth = [1.0, 1.08, 1.15, 1.28, 1.45, 1.62] if is_elite else [1.0, 1.05, 1.12, 1.18, 1.25, 1.35]
        projected = [st.session_state.price_val * g for g in growth]
        fig_forecast = px.area(x=years, y=projected, markers=True, color_discrete_sequence=['#50C878'])
        fig_forecast.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=20,b=0), xaxis_title="Projection Year", yaxis_title="Price (₹)")
        st.plotly_chart(fig_forecast, use_container_width=True)

with c_risk:
    st.markdown("<div class='section-header'>⚠️ Geo-Environmental Risk</div>", unsafe_allow_html=True)
    # Simulated environmental data
    risk_metrics = ['Flood Risk', 'AQI Index', 'Noise Level', 'Heat Island', 'Seismic']
    risk_values = [random.uniform(0.1, 0.4) if is_elite else random.uniform(0.3, 0.8) for _ in risk_metrics]
    
    fig_risk = go.Figure(data=go.Scatterpolar(r=risk_values, theta=risk_metrics, fill='toself', line_color='#9966cc'))
    fig_risk.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False, height=300, paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=40,r=40,t=20,b=20))
    st.plotly_chart(fig_risk, use_container_width=True)

with c_econ:
    st.markdown("<div class='section-header'>💎 Unit Economics</div>", unsafe_allow_html=True)
    if st.session_state.price_val:
        psf_final = st.session_state.price_val / sqft
        emi_est = (st.session_state.price_val * 0.8 * 0.09) / 12 
        
        # Investment Status logic
        status = "💎 STRONG BUY" if is_elite and sentiment_score > 80 else "⚖️ HOLD"
        
        st.metric("Price Per Sq.Ft", f"₹ {psf_final:,.0f}")
        st.metric("Est. Monthly EMI", f"₹ {format_indian_currency(emi_est)}")
        st.markdown(f"**Investment Signal:** `{status}`")
        
        econ_df = pd.DataFrame({
            "Metric": ["Rental Yield", "Tax Est. (1%)", "Liquidity Score"],
            "Value": ["3.2%", f"₹ {format_indian_currency(st.session_state.price_val*0.01)}", "High" if is_elite else "Moderate"]
        })
        st.table(econ_df)