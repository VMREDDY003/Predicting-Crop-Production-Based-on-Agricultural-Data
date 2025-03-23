import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open("../models/crop_production_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Title of the web app
st.title("Crop Production Prediction")

# Sidebar for user input
st.sidebar.header("Input Features")

def user_input_features():
    year = st.sidebar.number_input("Year", min_value=1960, max_value=2025, value=2020)
    area_harvested = st.sidebar.number_input("Area Harvested (in hectares)", min_value=0.0, value=1000.0)
    yield_ = st.sidebar.number_input("Yield (in kg/ha)", min_value=0.0, value=3000.0)
    data = {'Year': year,
            'Area_Harvested': area_harvested,
            'Yield': yield_}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display user input
st.subheader("User Input Features")
st.write(input_df)

# Prediction
prediction = model.predict(input_df)

# Display prediction
st.subheader("Predicted Crop Production (in tons)")
st.write(prediction[0])

# Additional information
st.markdown("""
## About
This application predicts crop production based on agricultural data such as the year, area harvested, and yield.
""")

st.markdown("""
## Business Use Cases
- Food Security and Planning
- Agricultural Policy Development
- Supply Chain Optimization
- Market Price Forecasting
- Precision Farming
- Agro-Technology Solutions
""")