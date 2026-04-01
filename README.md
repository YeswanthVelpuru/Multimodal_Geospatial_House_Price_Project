# 🏙️ Urban-AI: Multimodal Geospatial Deep Learning
### Fine-Grain Urban House Price Prediction Engine v2.4

Urban-AI is a cutting-edge valuation engine that leverages **Multimodal Deep Learning** and **Geospatial Graph Neural Networks (GNN)** to predict residential property values with block-level precision. By integrating structural attributes, satellite imagery, and urban micro-data (POI density, transit access, and environmental metrics), the system provides high-fidelity market insights.

---

## 🚀 Key Features
* **Multimodal Fusion:** Combines tabular property data with geospatial location intelligence.
* **Explainable AI (XAI):** Integrated **SHAP (Lundberg et al.)** waterfall plots to visualize feature contributions for every prediction.
* **Urban Micro-Data Extraction:** Analyzes Point-of-Interest (POI) clusters including hospitals, schools, and transit hubs via Nominatim API.
* **Interactive Digital Twin UI:** A high-performance Streamlit dashboard featuring Dark-Matter Mapbox visualizations and radar quality matrices.
* **Production-Ready Backend:** FastAPI server utilizing **ONNX Runtime** for high-speed model inference.

---

## 🛠️ Tech Stack
| Category | Tools |
| :--- | :--- |
| **Frontend** | Streamlit, Plotly, CSS3 (Glassmorphism) |
| **Backend** | FastAPI, Uvicorn, Pydantic |
| **Machine Learning** | Scikit-Learn, ONNX, SHAP |
| **Geospatial** | GeoPandas, Nominatim (OSM), Mapbox |
| **Experiment Tracking** | MLflow |

---

## 📁 Project Structure
```text
House_Price_Project/
├── app.py                # Streamlit Dashboard (Frontend)
├── api.py                # FastAPI Inference Engine (Backend)
├── model_training.py     # ML Pipeline & ONNX Export
├── data_processing.py    # Feature Engineering & Scaling
├── dataset/              # Raw and Processed CSV data
├── models/               # Saved .pkl and .onnx artifacts
├── scaler.pkl            # Trained StandardScaler object
└── requirements.txt      # Project Dependencies