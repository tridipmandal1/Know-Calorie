# app/predict.py

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def predict_calories(img_path, model_path='../model/food_model.h5'):
    """
    Predict the calorie content of a food image.
    """
    model = load_model(model_path)

    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predicted_calories = model.predict(img_array)[0][0]
    return predicted_calories
