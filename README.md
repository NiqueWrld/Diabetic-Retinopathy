# Diabetic Retina Diagnosis App

This application uses a pre-trained InceptionV3 model to classify diabetic retinopathy from retina images.

## Features
- Upload retina images for diagnosis.
- Get predictions with probabilities for each class.
- Pre-trained on the APTOS 2019 dataset.

## Setup
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Place the trained model (`best_model.keras`) in the root directory.
4. Run the app: `python run.py`.

## Usage
1. Open the app in a browser (default: `http://127.0.0.1:5000`).
2. Upload a retina image.
3. View the prediction and probabilities.