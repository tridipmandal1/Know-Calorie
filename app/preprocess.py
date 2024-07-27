# app/preprocess.py

import os
import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def is_image_file(filepath):
    try:
        Image.open(filepath).verify()
        return True
    except (IOError, SyntaxError) as e:
        print(f"Warning: {filepath} is not a valid image file.")
        return False

def load_data(image_dir, csv_file, img_height=224, img_width=224):
    """
    Load images and labels, and split into training and validation sets.
    """
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Debug: Print column names to ensure correct loading
    print(f"Columns in CSV file: {df.columns.tolist()}")

    if 'Food item' not in df.columns or 'Calories' not in df.columns:
        raise KeyError("CSV file must contain 'Food item' and 'Calories' columns")

    df['Food item'] = df['Food item'].str.lower()  # Ensure consistency in food item names

    # Create a dictionary to map food items to calorie values
    food_to_calories = dict(zip(df['Food item'], df['Calories']))

    # Create lists to store images and labels
    images = []
    labels = []
    missing_food_items = set()

    # Iterate over the directories in the image directory
    for food_item in os.listdir(image_dir):
        food_item_path = os.path.join(image_dir, food_item)
        if os.path.isdir(food_item_path):
            food_item_lower = food_item.lower()
            if food_item_lower not in food_to_calories:
                missing_food_items.add(food_item_lower)
                continue  # Skip this food item

            for img_name in os.listdir(food_item_path):
                img_path = os.path.join(food_item_path, img_name)
                if is_image_file(img_path):  # Check if the file is a valid image
                    images.append(img_path)
                    labels.append(food_to_calories[food_item_lower])

    # Debug: Print missing food items
    if missing_food_items:
        print(f"Missing food items in CSV: {missing_food_items}")

    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Split into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=pd.DataFrame({'filename': train_images, 'class': train_labels}),
        x_col='filename',
        y_col='class',
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='raw'
    )

    val_generator = val_datagen.flow_from_dataframe(
        dataframe=pd.DataFrame({'filename': val_images, 'class': val_labels}),
        x_col='filename',
        y_col='class',
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='raw'
    )

    return train_generator, val_generator
