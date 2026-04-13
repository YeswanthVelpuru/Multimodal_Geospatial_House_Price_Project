from PIL import Image
import numpy as np
import os

os.makedirs("data/images", exist_ok=True)

for i in range(1, 6):
    img = (np.random.rand(224, 224, 3) * 255).astype("uint8")
    Image.fromarray(img).save(f"data/images/house{i}.jpg")

print("Images generated!")