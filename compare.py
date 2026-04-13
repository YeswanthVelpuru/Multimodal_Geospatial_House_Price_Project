from evaluation.model_comparison import evaluate, compare_models

results = {}

# Example placeholders (replace with actual predictions)
results["ML (XGBoost)"] = evaluate(y_true, y_pred_ml)
results["DL (Tabular NN)"] = evaluate(y_true, y_pred_dl)
results["Multimodal (CNN+Tabular)"] = evaluate(y_true, y_pred_multi)

df = compare_models(results)
print(df)