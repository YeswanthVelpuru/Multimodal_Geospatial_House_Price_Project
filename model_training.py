import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import mlflow
from data_processing import get_processed_data

# --- Phase 4: Model Architecture ---
class MultimodalHousePredictor(nn.Module):
    def __init__(self, struct_dim=7, geo_dim=4):
        super(MultimodalHousePredictor, self).__init__()
        self.structural_branch = nn.Sequential(nn.Linear(struct_dim, 32), nn.ReLU())
        self.geospatial_branch = nn.Sequential(nn.Linear(geo_dim, 16), nn.ReLU())
        self.final_head = nn.Sequential(
            nn.Linear(32 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, struct_x, geo_x):
        x1 = self.structural_branch(struct_x)
        x2 = self.geospatial_branch(geo_x)
        combined = torch.cat((x1, x2), dim=1)
        return self.final_head(combined)

def run_training():
    # 1. Load synchronized data
    X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, feats = get_processed_data()

    # 2. Phase 3: Baseline (Random Forest)
    rf = RandomForestRegressor(n_estimators=50).fit(X_train_s, y_train)
    y_pred = rf.predict(X_val_s)
    mse = mean_squared_error(y_val, y_pred)
    
    print(f"Baseline R2 Score: {rf.score(X_val_s, y_val):.4f}")
    print(f"Validation MSE: {mse:.2f}")

    # 3. MLflow Logging
    with mlflow.start_run():
        mlflow.log_param("model_type", "Multimodal DL")
        mlflow.log_metric("mse", float(mse))

    # 4. Phase 5: Train Multimodal DL
    model = MultimodalHousePredictor(struct_dim=7, geo_dim=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    print("--- Phase 5: Training Multimodal Model ---")
    model.train()
    
    # Convert data to tensors once for efficiency
    s_t = torch.tensor(X_train_s[:, :7], dtype=torch.float32)
    g_t = torch.tensor(X_train_s[:, 7:], dtype=torch.float32)
    # Ensure y_t is handled correctly whether y_train is Series or Array
    y_vals = y_train.values if hasattr(y_train, 'values') else y_train
    y_t = torch.tensor(y_vals, dtype=torch.float32).view(-1, 1)

    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(s_t, g_t)
        loss = criterion(outputs, y_t)
        loss.backward()
        optimizer.step()
        
    print(f"Final Training Loss: {loss.item():.4f}")

    # 5. Phase 8: Package Model (State Dict)
    torch.save(model.state_dict(), "house_price_model.pth")
    print("Model Packaged as house_price_model.pth")

    # 6. PACKAGE MODEL (TorchScript Serialization)
    # We do this inside the function so it has access to the 'model' variable
    model.eval() # Set to evaluation mode before tracing
    example_input_struct = torch.randn(1, 7)
    example_input_geo = torch.randn(1, 4)

    traced_model = torch.jit.trace(model, (example_input_struct, example_input_geo))
    traced_model.save("house_price_model.pt")
    print("✅ TorchScript model saved as house_price_model.pt")

if __name__ == "__main__":
    run_training()