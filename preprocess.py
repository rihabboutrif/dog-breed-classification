from PIL import Image
import numpy as np
import config

def preprocess_image(image_file):
    img = Image.open(image_file).convert("RGB")
    img = img.resize((config.IMG_SIZE, config.IMG_SIZE))
    img_array = np.array(img) / 255.0  # normalize
    return img_array
