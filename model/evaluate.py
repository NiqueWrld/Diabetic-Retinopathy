import tensorflow as tf
import numpy as np
from .preprocess import preprocess_image
from .utils import focal_loss_fixed

# Load the trained model
model = tf.keras.models.load_model("best_model.keras", custom_objects={"focal_loss_fixed": focal_loss_fixed})

CLASSES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

def predict_image(image_path):
    """Predict the class of the uploaded image."""
    image = preprocess_image(image_path)
    predictions = model.predict(image)
    predicted_class = CLASSES[np.argmax(predictions)]
    probabilities = {CLASSES[i]: round(float(pred), 4) for i, pred in enumerate(predictions[0])}
    return predicted_class, probabilities