import cv2
import numpy as np

def preprocess_image(image_path):
    """Preprocess the image for model input."""
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize to 299x299 for InceptionV3
    image = cv2.resize(image, (299, 299))
    # Normalize pixel values
    image = image / 255.0
    # Expand dimensions to match model input
    image = np.expand_dims(image, axis=0)
    return image