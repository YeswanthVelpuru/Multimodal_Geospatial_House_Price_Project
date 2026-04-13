import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class HouseDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 🔥 IMAGE
        img_name = self.data.iloc[idx]['image']
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # 🔥 TABULAR (FIXED)
        features = self.data.drop(columns=["image", "price"])
        tabular = torch.tensor(features.iloc[idx].values.astype(float), dtype=torch.float32)

        # 🔥 TARGET
        price = torch.tensor(self.data.iloc[idx]['price'], dtype=torch.float32)

        return image, tabular, price