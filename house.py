import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib  # Import joblib for saving/loading models

# Load the dataset
df = pd.read_csv("C:\\Users\\Acer\\Downloads\\Housing.csv")

# Convert categorical features to numeric using OneHotEncoding
categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# Define features (X) and target variable (y)
X = df.drop(columns=['price'])
y = df['price']

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model and scaler
joblib.dump(model, 'house_price_model.pkl')  # Saving model
joblib.dump(scaler, 'scaler.pkl')  # Saving scaler

print("✅ Model and scaler saved successfully!")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📊 Mean Squared Error: {mse}")
print(f"📈 R-squared Score: {r2}")
# Calculate Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Convert to percentage
accuracy = 100 - mape  # Accuracy in percentage

print(f"📊 Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"✅ Model Prediction Accuracy: {accuracy:.2f}%")


# Visualize actual vs predicted prices
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, label="Predictions")
plt.plot(y_test, y_test, color='red', linestyle='dashed', label="Perfect Fit")  # y=x line
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices with Best Fit Reference")
plt.legend()
plt.show()
