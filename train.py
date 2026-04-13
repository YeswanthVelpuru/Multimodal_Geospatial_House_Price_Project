import pandas as pd
import cv2
import numpy as np
from cnn_model import create_cnn

# Load CSV
data = pd.read_csv("data/house_data.csv")

images = []
prices = []

for i, row in data.iterrows():
    img_path = f"data/images/{row['image_name']}"
    
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128,128))
    
    images.append(img)
    prices.append(row['price'])

X = np.array(images) / 255.0
y = np.array(prices)

# Train CNN
model = create_cnn()
model.fit(X, y, epochs=5)