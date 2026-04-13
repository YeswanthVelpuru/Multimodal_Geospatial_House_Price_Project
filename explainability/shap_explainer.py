import shap
import torch

class ShapExplainer:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def predict(self, x):
        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            return self.model.tabular(x_tensor).numpy()

    def explain(self, sample):
        explainer = shap.KernelExplainer(self.predict, sample)
        shap_values = explainer.shap_values(sample)
        return shap_values