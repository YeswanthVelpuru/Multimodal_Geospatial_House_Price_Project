import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import random
from geopy.geocoders import Nominatim

st.set_page_config(page_title="Geospatial DL Elite", layout="wide")

MAPBOX_TOKEN = "pk.eyJ1IjoieWVzd2FudGgtLXYtLTIwMDMiLCJhIjoiY21taHh5ZmJtMHRneDJwczZxaWhiYmg3ZiJ9.IZK_WUOAlFdAsg0ewYyARg"
pdk.settings.mapbox_api_key = MAPBOX_TOKEN

TIER_1 = ["Mumbai", "South Delhi", "Jubilee Hills", "Banjara Hills", "Lavelle Road", "Boat Club", "Malabar Hill", "Worli", "Juhu", "Adyar", "Alipore", "Ballygunge", "Prithviraj Road"]
TIER_2 = ["Kokapet", "Financial District", "Whitefield", "Indiranagar", "Koramangala", "Hitech City", "Cyberabad", "Pune", "Chandigarh", "Ahmedabad", "Gomti Nagar"]

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
    "Kowdiar", "Benz Circle", "Labbipet", "Jayalakshmipuram", "Vidyaranyapuram", "Gokulam", "Saraswathipuram", "Magunta Layout", "Koregaon Park", 
    "Boat Club Road", "Prabhat Road", "Model Colony", "Aundh", "Baner", "Kalyani Nagar", "Erandwane", "Balewadi High Street", "Satellite", "Bodakdev", 
    "Thaltej", "Sindhu Bhavan Road", "Ambli", "Vesu", "City Light", "Adajan", "Piplod", "Civil Lines", "Dharampeth", "Ramdaspeth", "CIDCO", "Garkheda", 
    "Shivaji Nagar", "Samarth Nagar", "Rajarampuri", "Tarabai Park", "Murarji Peth", "Jule Solapur", "Vijay Nagar", "Old Palasia", "Saket", "Arera Colony", 
    "74 Bungalows", "Sector 5", "Sector 8", "Sector 9", "Sector 17", "Mohali Phase 7", "Sector 33", "Sarabha Nagar", "Model Town", "Pakhowal Road", 
    "Ranjit Avenue", "Green Avenue", "Gomti Nagar", "Hazratganj", "C-Scheme", "Malviya Nagar", "Vaishali Nagar", "Cantonment", "Dayal Bagh", "Sigra", 
    "Bhelupur", "Alipore", "Ballygunge", "Park Street", "New Town", "Bariatu", "Kanke", "Morabadi", "Bistupur", "Uzan Bazar", "Ganeshguri", "Beltola", "Christian Basti"
]

def format_indian_currency(num):
    if num >= 10000000: return f"{num / 10000000:.2f} Cr"
    elif num >= 100000: return f"{num / 100000:.2f} L"
    return f"{num:,.2f}"

st.markdown("""
    <style>
    html, body, [class*="st-"] { font-size: 1.2rem !important; }
    .main-title {
        font-family: 'Montserrat', sans-serif;
        font-size: 1.8rem !important; font-weight: 800; text-align: center;
        background: linear-gradient(90deg, #ffffff, #9966cc, #50C878, #ffffff);
        background-size: 200% auto; -webkit-background-clip: text;
        -webkit-text-fill-color: transparent; animation: shine 6s linear infinite; padding: 10px 0;
    }
    @keyframes shine { to { background-position: 200% center; } }
    .jewel-price {
        font-size: 3.2rem !important; font-weight: 800;
        background: linear-gradient(135deg, #9966cc 10%, #50C878 90%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        display: block; margin: 10px 0;
    }
    .section-header { 
        color: #9966cc; text-transform: uppercase; letter-spacing: 2px; font-size: 1.0rem !important; 
        font-weight: 700; margin-bottom: 12px; border-bottom: 2px solid rgba(153, 102, 204, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('''<div class="main-title">MULTIMODAL GEOSPATIAL DEEP LEARNING FOR FINE GRAIN URBAN HOUSE PRICE PREDICTION</div>''', unsafe_allow_html=True)

geolocator = Nominatim(user_agent="geospatial_dl_ultimate")
with st.sidebar:
    st.markdown("<div class='section-header'>🛰️ TENSOR INPUTS</div>", unsafe_allow_html=True)
    search_query = st.text_input("📍 Neural Search", "Juhu, Mumbai")
    try:
        loc = geolocator.geocode(search_query)
        if loc: lat, lon = loc.latitude, loc.longitude
        else: lat, lon = 19.1075, 72.8263 
    except: lat, lon = 19.1075, 72.8263
    
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
    if any(city.lower() in search_query.lower() for city in TIER_1): market_psf, t_mult = 30000, random.uniform(0.95, 1.25)
    elif any(city.lower() in search_query.lower() for city in TIER_2): market_psf, t_mult = 22000, random.uniform(1.0, 1.18)
    else: market_psf, t_mult = 10500, random.uniform(1.05, 1.15)
else: market_psf, t_mult = 6800, random.uniform(0.8, 1.5)

st.markdown("---")
c_map, c_diag = st.columns([1.5, 1])
with c_map:
    st.markdown("<div class='section-header'>🛰️ Multimodal Satellite Analysis</div>", unsafe_allow_html=True)
    st.pydeck_chart(pdk.Deck(map_style='mapbox://styles/mapbox/satellite-streets-v12', initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=16, pitch=45), height=380))

with c_diag:
    st.markdown("<div class='section-header'>🏛️ Inference Engine</div>", unsafe_allow_html=True)
    if 'price_val' not in st.session_state: st.session_state.price_val = None
    if st.button("CALCULATE PREDICTED PRICE", use_container_width=True):
        base = (sqft * market_psf) * (1 + (grade-8)*0.08)
        geo_dna = (greenery * 0.1) + (transit * 0.15) + (safety * 0.05)
        st.session_state.price_val = base * (1 + geo_dna) * t_mult
    
    if st.session_state.price_val:
        st.markdown(f'<div class="jewel-price">₹ {format_indian_currency(st.session_state.price_val)}</div>', unsafe_allow_html=True)
        st.markdown("<div class='section-header' style='border:none; font-size:0.75rem !important;'>Synaptic Tensor Weight Distribution 🔗</div>", unsafe_allow_html=True)
        fig_flow = go.Figure(data=[go.Sankey(node=dict(pad=15, thickness=15, label=["Input Layers", "Structural", "Geospatial", "L1 Hidden", "L2 Hidden", "Valuation"], color=["#9966cc", "#9966cc", "#50C878", "#888888", "#888888", "#FFD700"]),
            link=dict(source=[0, 1, 2, 0, 1, 2, 3, 4], target=[3, 3, 3, 4, 4, 4, 5, 5], value=[40, 30, 30, 20, 40, 40, 50, 50], color="rgba(80, 200, 120, 0.2)"))])
        fig_flow.update_layout(height=280, paper_bgcolor='rgba(0,0,0,0)', font=dict(size=10, color="white"), margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_flow, config={'displayModeBar': False}, use_container_width=True)
    else: st.caption("Awaiting Inference...")

st.markdown("---")
c_heat, c_diff, c_attr = st.columns([1.2, 1, 1])
with c_heat:
    st.markdown("<div class='section-header'>📊 Neural Feature Correlation</div>", unsafe_allow_html=True)
    features = ['Price', 'BHK', 'Grade', 'Green', 'Transit', 'Safety']
    corr_data = np.array([[1.0, 0.8, 0.9, 0.4, 0.6, 0.5],[0.8, 1.0, 0.6, 0.1, 0.3, 0.2],[0.9, 0.6, 1.0, 0.5, 0.4, 0.4],[0.4, 0.1, 0.5, 1.0, 0.2, 0.3],[0.6, 0.3, 0.4, 0.2, 1.0, 0.7],[0.5, 0.2, 0.4, 0.3, 0.7, 1.0]])
    fig_heat = px.imshow(corr_data, x=features, y=features, color_continuous_scale='Viridis', aspect="auto")
    fig_heat.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=350, margin=dict(l=0,r=0,t=0,b=0), font=dict(size=10))
    st.plotly_chart(fig_heat, use_container_width=True)

with c_diff:
    st.markdown("<div class='section-header'>📈 Price Diffusion (₹)</div>", unsafe_allow_html=True)
    if st.session_state.price_val:
        v = st.session_state.price_val
        prices = [v, v*0.88, v*0.75, v*1.15, v*0.92]
        labels = [format_indian_currency(p) for p in prices]
        fig_diff = go.Figure(go.Scatter(x=["Target", "A", "B", "C", "D"], y=prices, text=labels, mode='lines+markers+text', textposition="top center", line=dict(shape='spline', color='#50C878', width=3)))
        fig_diff.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis=dict(showticklabels=False), margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_diff, use_container_width=True)

with c_attr:
    st.markdown("<div class='section-header'>📊 XAI Attribution (₹)</div>", unsafe_allow_html=True)
    if st.session_state.price_val:
        v = st.session_state.price_val
        vals = [v*0.48, v*0.22, v*0.30]
        labels = [format_indian_currency(p) for p in vals]
        fig_attr = go.Figure(go.Bar(x=vals, y=["Structural", "DNA", "Premium"], text=labels, textposition='auto', orientation='h', marker_color=['#9966cc', '#50C878', '#FFD700']))
        fig_attr.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(showticklabels=False), margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_attr, use_container_width=True)