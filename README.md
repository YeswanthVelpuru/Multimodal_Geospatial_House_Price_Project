🛰️ Multimodal Geospatial DL: Fine-Grain Urban House Price Prediction

A high-fidelity **Neural Inference Dashboard** designed to predict house prices in the Indian market with "fine-grain" precision. By fusing structural architectural data with geospatial "DNA" layers, the model provides hyper-localized valuations across major Indian metros.


💎 Core Features

🧠 Neural Inference Engine
* **Multimodal Fusion:** Integrates structural features (BHK, SqFt, Arch. Grade) with geospatial layers (NDVI Greenery, Transit Accessibility, Safety Indices).
* **Synaptic Tensor Flow:** A custom **Sankey-based visualization** illustrating the distribution of 25,000 weights across neural layers during inference.
* **XAI (Explainable AI):** Real-time attribution analysis identifying value derivation from the **Structural Core**, **Geospatial DNA**, and **Elite Zone Premiums**.

 🇮🇳 Indian Market Intelligence
* **Tiered Elite Registry:** A proprietary database of 100+ high-value neighborhoods (Juhu, Adyar, Jubilee Hills) calibrated with specific PSF benchmarks.
* **Indian Denomination Logic:** Built-in currency helper scaling valuations into **Lakhs (L)** and **Crores (Cr)**.
* **Dynamic Neural Search:** Integrated Geocoding via Nominatim for real-time satellite positioning.


🏗️ Project Artifacts & Methodology

The "Fine-Grain" precision of this engine is derived from high-dimensional latent artifacts:

1.  **📂 Neural Weight Tensors (`model_weights.bin`)**
    * **Dimension:** 25,000 Active Parameters.
    * **Function:** Defines non-linear relationships between urban infrastructure and structural quality.
2.  **🗺️ Elite Registry Atlas (`elite_registry.json`)**
    * **Coverage:** 100+ Premium Indian Micro-markets.
    * **Benchmarks:** Tier 1 (₹30k), Tier 2 (₹22k), Tier 3 (₹10.5k).
3.  **🛰️ Geospatial DNA Vectors (`dna_tensors.csv`)**
    * **Feature Set:** 60 Hidden "Urban DNA" vectors derived from satellite morphology telemetry.

## 📊 Model Architecture Overview
The inference process follows a strict **Multimodal Feed-Forward** pipeline:
1.  **Input Layer:** Structural (BHK/SqFt) + Geospatial (NDVI/Safety).
2.  **Hidden Layer 1 (L1):** Feature Cross-Correlation (Linear + ReLU).
3.  **Hidden Layer 2 (L2):** Latent Vector Fusion (**25k Tensors**).
4.  **Output Head:** Valuation Synthesis with Indian Denomination Scaling.

## 🛠️ Technical Stack
* **Frontend:** Streamlit (Custom Jewel-Tone CSS)
* **Geospatial:** PyDeck (Mapbox GL), Geopy (Nominatim)
* **Analytics:** Plotly Graph Objects (Sankey, Heatmaps, XAI)
* **Backend:** FastAPI (Prediction Endpoint)

## 🚀 Installation & Setup

```bash
# 1. Clone the Repository
git clone [https://github.com/YeswanthVelpuru/Multimodal_Geospatial_House_Price_Project.git](https://github.com/YeswanthVelpuru/Multimodal_Geospatial_House_Price_Project.git)
cd Multimodal_Geospatial_House_Price_Project

# 2. Install Dependencies
pip install streamlit pandas pydeck plotly numpy geopy fastapi uvicorn pydantic

# 3. Run the Application
streamlit run app.py

📊 Feature Documentation
Category,Features,Impact on Valuation
Structural,"BHK, SqFt, Arch. Grade, Smart Automation",Primary Base Value
Geospatial,"NDVI Index, Transit Node Proximity",DNA Multiplier
Zone,"Tier 1 (Metros), Tier 2 (Emerging), Tier 3 (Elite)",PSF Baseline Calibration
Safety,Safety Perception Index,Sentiment-based Premium

🧪 Testing & Validation
Pytest Suite (test_api.py): Validates the FastAPI /predict schema against 422 errors.
Flake8 Linting: Ensures PEP8 compliance and code stability.
XAI SHAP Values: Integrated "Red/Green" attribution logic for model transparency.
