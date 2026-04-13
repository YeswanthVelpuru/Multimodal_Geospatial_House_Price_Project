import pandas as pd
import cv2
import numpy as np
from cnn_model import create_cnn

# Load CSV
data = pd.read_csv("data/house_data.csv")

images = []
prices = []

for i, row in data.iterrows():
    img_name = row.get('image_name', 'img1.jpg')

    if pd.isna(img_name):
        img_name = 'img1.jpg'

    img_path = f"data/images/{img_name}"
    img = cv2.imread(img_path)

    if img is None:
        img = np.zeros((128,128,3))  # fallback
    else:
        img = cv2.resize(img, (128,128))

    images.append(img)

    # make sure 'price' column exists
    prices.append(row['price'])

# Convert to arrays
X = np.array(images) / 255.0
y = np.array(prices)

# Train CNN
model = create_cnn()
model.fit(X, y, epochs=5)