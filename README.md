# 🛰️ Multimodal Geospatial House Price Prediction
**A Fine-Grain Urban Intelligence Engine using Deep Learning**

---

### 📝 Project Abstract
This research project implements a **Multimodal Deep Learning Architecture** designed for high-precision urban real estate valuation. Unlike traditional models that rely solely on structural attributes, this engine treats geographical coordinates as a **latent feature space**. By fusing structural property tensors with satellite-derived environmental micro-data (NDVI, AQI, and Urban Density), the system achieves a fine-grain understanding of market dynamics. The framework is built with **Explainable AI (XAI)** principles, utilizing SHAP values to provide transparency in AI-driven financial decisions.



### 🏗️ System Architecture & Data Fusion
The engine utilizes a **Triple-Branch Fusion** approach to ensure accuracy:

1. **Structural Tensor Branch**: Processes `[BHK, SqFt, Grade]` through a normalized dense layer.
2. **Geospatial DNA Branch**: Decodes `[Lat, Lon]` into environmental vectors (Greenery, Transit, Safety).
3. **Architectural State Branch**: Adjusts for `[Condition, Age]` to calculate depreciation scalars.

### 🚀 Technical Milestones
- **Automated CI/CD**: Verified builds on Python 3.9, 3.10, and 3.11 via GitHub Actions.
- **Inference Security**: Optimized for PyTorch 2.6+ with secure weight loading.
- **Explainable AI**: Integrated SHAP Waterfall plots for real-time feature impact analysis.
