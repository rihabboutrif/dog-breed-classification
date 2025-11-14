import tensorflow as tf
import numpy as np
import config
from model_loader import load_my_model

# Load model once
model = load_my_model()

def predict_breed(img_array):
    # Predict probabilities for each class
    prediction = model.predict(np.expand_dims(img_array, axis=0))
    
    # Get index of the highest probability
    class_idx = np.argmax(prediction)
    
    # Get the actual confidence (highest probability)
    confidence = float(np.max(prediction))
    
    # Get the corresponding class name
    breed = config.CLASS_NAMES[class_idx]
    
    return breed, confidence
