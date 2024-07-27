# app/model.py

import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from preprocess import load_data

def build_model(input_shape=(224, 224, 3)):
    """
    Build a CNN model for calorie prediction.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='linear')  # Regression output
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error', metrics=['mae'])
    return model

def train_model():
    """
    Train the CNN model for calorie prediction.
    """
    image_dir = 'dataset/food_images'
    csv_file = 'dataset/food_calories.csv'
    train_generator, val_generator = load_data(image_dir, csv_file)

    model = build_model()
    
    print("Training the model...")
    model.fit(
        train_generator,
        epochs=20,
        validation_data=val_generator
    )

    os.makedirs('model', exist_ok=True)
    model.save('model/food_model.h5')
    print("Model training completed and saved as 'food_model.h5'.")

if __name__ == "__main__":
    train_model()
