import joblib
import numpy as np
import pandas as pd

# Load the saved model and scaler
model = joblib.load('house_price_model.pkl')
scaler = joblib.load('scaler.pkl')

# Mean Absolute Percentage Error (MAPE) from training (adjust based on your results)
mape = 21.04  # Example MAPE from training (modify if needed)
accuracy = 100 - mape  # Prediction Accuracy

def ask_house_price():
    print("\nEnter house details to predict the price:")

    # Get user input
    area = float(input("Enter area (sq ft): "))
    bedrooms = int(input("Number of bedrooms: "))
    bathrooms = int(input("Number of bathrooms: "))
    stories = int(input("Number of stories: "))
    parking = int(input("Number of parking spaces: "))

    # Get user input for categorical features (Yes/No)
    mainroad = int(input("Is it on the main road? (1 for Yes, 0 for No): "))
    guestroom = int(input("Does it have a guestroom? (1 for Yes, 0 for No): "))
    basement = int(input("Does it have a basement? (1 for Yes, 0 for No): "))
    hotwaterheating = int(input("Does it have hot water heating? (1 for Yes, 0 for No): "))
    airconditioning = int(input("Does it have air conditioning? (1 for Yes, 0 for No): "))
    prefarea = int(input("Is it in a preferred area? (1 for Yes, 0 for No): "))

    # Get furnishing status (One-hot encoding)
    furnishing = input("Furnishing status (furnished, semi-furnished, unfurnished): ").strip().lower()

    # Convert inputs into the format used in training
    new_house = {
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'stories': stories,
        'parking': parking,
        'mainroad_yes': mainroad,
        'guestroom_yes': guestroom,
        'basement_yes': basement,
        'hotwaterheating_yes': hotwaterheating,
        'airconditioning_yes': airconditioning,
        'prefarea_yes': prefarea,
        'furnishingstatus_semi-furnished': 1 if furnishing == 'semi-furnished' else 0,
        'furnishingstatus_unfurnished': 1 if furnishing == 'unfurnished' else 0
    }

    # Convert to DataFrame
    new_house_df = pd.DataFrame([new_house])

    # Scale input
    new_house_scaled = scaler.transform(new_house_df)

    # Predict house price
    predicted_price = model.predict(new_house_scaled)

    # Display the result
    print(f"\n🔹 Predicted House Price: ₹{predicted_price[0]:,.2f}")
    print(f"✅ Prediction Accuracy: {accuracy:.2f}%\n")

# Run the function
ask_house_price()
