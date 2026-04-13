import torch
from torch.utils.data import DataLoader, random_split
from models.multimodal_model import MultiModalModel
from data.dataset import HouseDataset
import torchvision.transforms as transforms

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------- DATASET ----------------
dataset = HouseDataset("data/data.csv", "data/images", transform)

# 🔥 TRAIN / VALIDATION SPLIT
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 🔥 DATALOADERS
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# ---------------- MODEL ----------------
model = MultiModalModel(tabular_input_size=3)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 🔥 LOSS TRACKING
train_losses = []

# ---------------- TRAINING ----------------
for epoch in range(10):
    model.train()
    epoch_loss = 0

    for img, tab, price in train_loader:
        output = model(img, tab)
        loss = criterion(output.squeeze(), price)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    print(f"Epoch {epoch} Loss: {avg_loss:.4f}")

# ---------------- SAVE MODEL ----------------
torch.save(model.state_dict(), "model.pth")

# ---------------- VALIDATION ----------------
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for img, tab, price in val_loader:
        output = model(img, tab)

        # 🔥 FIXED (no 0-d error)
        y_true.extend(price.view(-1).numpy())
        y_pred.extend(output.view(-1).numpy())

# ---------------- METRICS ----------------
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("\n📊 MODEL PERFORMANCE (Validation Set):")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R2 Score: {r2:.4f}")

# ---------------- LOSS GRAPH ----------------
plt.plot(train_losses)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.show()