# app/calorie_info.py

import pandas as pd

# Load the calorie data
calorie_data = pd.read_csv('dataset/food_calories.csv')

def get_calories(food_name):
    """
    Get the calories for the given food item.
    Args:
    food_name (str): Name of the food item.

    Returns:
    int: Calories of the food item.
    """
    row = calorie_data[calorie_data['food_name'] == food_name]
    if not row.empty:
        quantity = row['quantity'].values[0]
        calories = row['calories'].values[0]
        return calories, quantity
    return None, None
