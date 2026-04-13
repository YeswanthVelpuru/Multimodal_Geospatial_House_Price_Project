import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "R2": r2}


# 🔥 Dummy data (replace with real predictions later)
y_true = np.array([100, 200, 300, 400])

y_pred_ml = np.array([110, 210, 290, 410])
y_pred_dl = np.array([105, 195, 310, 390])
y_pred_multi = np.array([102, 198, 305, 402])

results = {}

results["ML (XGBoost)"] = evaluate(y_true, y_pred_ml)
results["DL (Tabular NN)"] = evaluate(y_true, y_pred_dl)
results["Multimodal (CNN+Tabular)"] = evaluate(y_true, y_pred_multi)

print(results)