import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from data_preprocessing import load_and_preprocess_data

# Load and preprocess data
df_pivot = load_and_preprocess_data("c:/Users/VMReddy/Desktop/crop-prediction/data/FAOSTAT_data.xlsx")

# Feature Selection
X = df_pivot[['Year', 'Area_Harvested', 'Yield']]
y = df_pivot['Production']

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Models
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluation Metrics
print("\nRandom Forest Performance:")
print(f"MAE: {mean_absolute_error(y_test, y_pred_rf):.2f}")
print(f"MSE: {mean_squared_error(y_test, y_pred_rf):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred_rf):.2f}")

# Save the best model
with open("c:/Users/VMReddy/Desktop/crop-prediction/models/crop_production_model.pkl", "wb") as model_file:
    pickle.dump(rf_model, model_file)

print("\nBest Model (Random Forest) Saved Successfully!")