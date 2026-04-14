🛰️ Multimodal Geospatial Deep Learning for Fine-Grain Urban House Price Prediction

A production-grade AI system that predicts house prices using Multimodal Deep Learning + Geospatial Intelligence + Market Tier Modeling.

This platform combines:

🧠 Deep Learning (CNN + Tabular Fusion)
🌍 Geospatial Reachability
🇮🇳 Indian Real Estate Market Intelligence
to deliver hyper-local, explainable property valuations.

🚀 🔥 Core Features
🧠 Deep Learning Inference Engine
Multimodal Architecture: CNN (image) + Tabular features
Feature Fusion using Dense Neural Layers
Handles both structural + visual signals

🌍 Geospatial Intelligence
📍 Live Map Rendering (PyDeck + Mapbox)
🛰️ Isochrone Reachability Simulation
10-min walk radius
15-min drive radius
Location-aware valuation
🇮🇳 10-Tier City Intelligence System
Dynamic classification of Indian cities:
| Tier      | Multiplier  | Description     |
| --------- | ----------- | --------------- |
| Tier 1    | 4.0x        | Premium metros  |
| Tier 2    | 3.2x        | Emerging metros |
| Tier 3–5  | 2.6x – 1.9x | Growth cities   |
| Tier 6–8  | 1.7x – 1.3x | Developing      |
| Tier 9–10 | 1.2x – 1.1x | Low liquidity   |

✔ Automatically applied based on location input
✔ Real-world pricing simulation

🧠 AI Explainability (Stable XAI)

Instead of unstable SHAP for multimodal models:

✔ Feature Contribution Visualization
✔ BHK vs Area vs Structure importance
✔ Interpretable surrogate explanation
📊 Advanced Analytics Dashboard

Below-map intelligence layer includes:

💰 Price Breakdown (Base + Premium + Growth)
📈 5-Year ROI Projection
🏙️ Area Intelligence Radar (Safety, Greenery, Transit)
📊 Market Trends Simulation
🧠 AI Investment Recommendations
🧠 Smart AI Recommendation Engine

Dynamic decision support based on:

City Tier
Safety score
Transit accessibility

Example outputs:

🚀 “Premium investment zone”
📈 “Emerging growth market”
⚠️ “High-risk low-liquidity zone”

🏗️ System Architecture:
Tabular Data (BHK, SqFt, Location)
            │
            ▼
   Feature Processing
            │
            ▼
   Tabular Neural Network
            │

Image Data (House Images)
            │
            ▼
      CNN (ResNet18)
            │

      Feature Fusion
 (Concatenation Layer)
            │
            ▼
   Fully Connected Layers
            │
            ▼
  Final Price Prediction

📊 Model Architecture
CNN Backbone: ResNet18
Tabular Network: Fully Connected Layers
Fusion: Concatenation + Dense Layers
Loss: Mean Squared Error (MSE)

📈 Model Evaluation Metrics
| Metric   | Description        |
| -------- | ------------------ |
| RMSE     | Error magnitude    |
| MAE      | Absolute deviation |
| R² Score | Model accuracy     |

🧪 Model Comparison
| Model             | RMSE   | R²         |
| ----------------- | ------ | ---------- |
| ML (RandomForest) | High   | Medium     |
| Deep Learning     | Medium | High       |
| Geospatial DL     | Lowest | Highest 🚀 |

🛠️ Tech Stack
🔹 Frontend
Streamlit (Interactive Dashboard)
🔹 Deep Learning
PyTorch
Torchvision (ResNet18)
🔹 Geospatial
PyDeck (Mapbox)
Geopy
🔹 Visualization
Plotly
Radar, Bar, ROI graphs
🔹 Backend (Optional)
FastAPI
🔹 CI/CD
GitHub Actions
flake8 (lint)
pytest (testing)

Multimodal_Geospatial_House_Price_Project/
│
├── .github/
│   └── workflows/
│       └── python-package.yml
│
├── data/
│   ├── __init__.py
│   ├── data.csv
│   ├── house_data.csv
│   ├── dataset.py
│   ├── generate_images.py
│   │
│   ├── images/
│   │   ├── house1.jpg
│   │   ├── house2.jpg
│   │   ├── house3.jpg
│   │   ├── house4.jpg
│   │   └── house5.jpg
│   │
│   └── (additional images folder)
│       ├── house1.jpg
│       ├── house2.jpg
│       └── house3.jpg
│
├── evaluation/
│   ├── __init__.py
│   └── model_comparison.py
│
├── explainability/
│   ├── __init__.py
│   └── shap_explainer.py
│
├── mlruns/
│   └── 0/
│       └── models/
│           ├── m-6878ea32100d48709988560d67a9beb4/
│           │   └── artifacts/
│           │       ├── MLmodel
│           │       ├── conda.yaml
│           │       ├── model.pkl
│           │       ├── python_env.yaml
│           │       └── requirements.txt
│           │
│           ├── m-6fd1d7d3c8a14a5184540b7b6351d72f/
│           │   └── artifacts/
│           │       ├── MLmodel
│           │       ├── conda.yaml
│           │       ├── model.pkl
│           │       ├── python_env.yaml
│           │       └── requirements.txt
│           │
│           └── m-ae33271ca51149d788e2ec78f5079c7e/
│               └── artifacts/
│                   ├── MLmodel
│                   ├── conda.yaml
│                   ├── model.pkl
│                   ├── python_env.yaml
│                   └── requirements.txt
│
├── models/
│   ├── __init__.py
│   ├── cnn_model.py
│   ├── multimodal_model.py
│   ├── tabular_model.py
│   └── house_model.pkl
│
├── api.py
├── app.py
├── cnn_model.py
├── compare.py
├── data_processing.py
├── graph_features.py
├── image_features.py
├── market_features.py
├── model_training.py
├── rl_price_trend.py
├── setup_data.py
├── test_api.py
├── train.py
│
├── house_price_model.onnx
├── house_price_model.onnx.data
├── house_price_model.pt
├── model.onnx
├── model.pth
├── scaler.pkl
│
├── mlflow.db
├── requirements.txt
├── runtime.txt
├── README.md
├── .gitignore



🚀 Installation & Setup
# Clone repository
git clone https://github.com/YeswanthVelpuru/Multimodal_Geospatial_House_Price_Project.git

cd Multimodal_Geospatial_House_Price_Project

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
python train.py

🌐 Deployment
Streamlit Cloud ready
GitHub Actions CI integrated
Mapbox API enabled

⚠️ Note : Dataset and images are excluded to keep repository lightweight.

🧠 Key Innovation : “Fusion of Multimodal Deep Learning with Geospatial Reachability and Tier-Based Market Intelligence for fine-grain urban price prediction.”
