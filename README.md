🛰️ Multimodal Geospatial DL: Fine-Grain Urban House Price Prediction

A high-fidelity **Neural Inference Dashboard** designed to predict house prices in the Indian market with "fine-grain" precision. By fusing structural architectural data with geospatial "Reachability" layers, the model provides hyper-localized valuations across major Indian metros.

💎 Core Features

🧠 Neural Inference & Explainability
* **Reachability Mapping:** Replaced legacy heatmaps with **Isochrone Connectivity Circles** (10-min walk vs. 15-min drive) to simulate real-world urban accessibility.
* **XAI (Explainable AI):** Integrated **Neural Attention Weights** (SHAP-style) that visualize exactly how the model prioritizes Location DNA vs. Structural Core for every specific query.
* **Neighborhood Pulse:** Real-time **Radar Analytics** measuring Safety, AQI, Noise, Greenery, and Transit Node density.

🇮🇳 Indian Market Intelligence
* **Tiered Elite Registry:** A proprietary database of 100+ high-value neighborhoods (Juhu, Adyar, Jubilee Hills) calibrated with specific PSF benchmarks.
* **Dynamic Amenity Logic:** A context-aware generator that suggests high-value add-ons (EV Charging, Infinity Pools) based on the specific socio-economic tier of the location.
* **Indian Denomination Logic:** Built-in currency helper scaling valuations into **Lakhs (L)** and **Crores (Cr)**.

🏗️ Project Artifacts & Methodology

The "Fine-Grain" precision of this engine is derived from high-dimensional latent artifacts:

1. **📂 Neural Weight Tensors (`model_weights.bin`)**
   * **Dimension:** 25,000 Active Parameters.
   * **Function:** Defines non-linear relationships between urban infrastructure and structural quality.

2. **🗺️ Elite Registry Atlas (`elite_registry.json`)**
   * **Coverage:** 100+ Premium Indian Micro-markets.
   * **Benchmarks:** Tier 1 (₹30k), Tier 2 (₹22k), Tier 3 (₹10.5k).

3. **🛰️ Connectivity Vectors**
   * **Function:** Simulates "Reachability" using geospatial coordinates to calculate walkability and transit-oriented development (TOD) premiums.

Multimodal House Price Prediction System Architecture 

                ┌──────────────────────────┐
                │      Tabular Data        │
                │ (house_data.csv file)   │
                │ Features:               │
                │ - Bedrooms              │
                │ - Area (sqft)           │
                │ - Location              │
                │ - Price (Target)        │
                └────────────┬────────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │   Feature Processing     │
                │ (Scaling / Cleaning)     │
                └────────────┬────────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │  Tabular Feature Vector  │
                └────────────┬────────────┘


                ┌──────────────────────────┐
                │       Image Data         │
                │  (data/images/*.jpg)     │
                │ via image_name column    │
                └────────────┬────────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │      CNN Model           │
                │ Conv → Pool → Dense      │
                │ Extract Image Features   │
                └────────────┬────────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │   Image Feature Vector   │
                └────────────┬────────────┘


                    ┌────────▼────────┐
                    │  Feature Fusion │
                    │ (Concatenation) │
                    └────────┬────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │   Fully Connected NN     │
                │   (Dense Layers)         │
                └────────────┬────────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │ Predicted House Price    │
                └──────────────────────────┘

    
📊 Model Architecture Overview
The inference process follows a strict **Multimodal Feed-Forward** pipeline:
1. **Input Layer:** Structural (BHK/SqFt) + Geospatial (NDVI/Safety/Transit).
2. **Hidden Layer 1 (L1):** Feature Cross-Correlation (Linear + ReLU).
3. **Hidden Layer 2 (L2):** Latent Vector Fusion (**25k Tensors**).
4. **Output Head:** Valuation Synthesis with Indian Denomination Scaling.

🛠️ Technical Stack
* **Frontend:** Streamlit (Jewel-Tone Montserrat UI)
* **Geospatial:** PyDeck (Mapbox Navigation-Night Engine), Geopy
* **Analytics:** Plotly Graph Objects (Polar Radars, SHAP Bar Charts)
* **Backend:** FastAPI (Prediction Endpoint)

 🚀 Installation & Setup

```bash
# 1. Clone the Repository
git clone [https://github.com/YeswanthVelpuru/Multimodal_Geospatial_House_Price_Project.git](https://github.com/YeswanthVelpuru/Multimodal_Geospatial_House_Price_Project.git)
cd Multimodal_Geospatial_House_Price_Project

# 2. Install Dependencies
pip install streamlit pandas pydeck plotly numpy geopy fastapi uvicorn pydantic

# 3. Run the Application
streamlit run app.py

Category,Features,Impact on Valuation
Structural,"BHK, SqFt, Arch. Grade, Smart Core",Primary Base Value
Geospatial,"NDVI Index, Transit Node Proximity",DNA Multiplier
Reachability,10-min Walk / 15-min Drive Radius,Accessibility Premium
Investment,"Rental Yield, Liquidity Score",Financial Sentiment

## Deep Learning Module

- CNN model implemented using TensorFlow
- Processes image data from data/images/
- Handles missing images using fallback mechanism
- Trained using train.py
