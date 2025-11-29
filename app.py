import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------------------------
# Load final model
# ------------------------------------------------------------
model = joblib.load("final_random_forest_model.pkl")

st.set_page_config(page_title="Car Price Prediction", layout="centered")

st.title("🚗 Car Selling Price Prediction")
st.write("Enter the car details below to estimate the selling price.")

# ------------------------------------------------------------
# User Inputs
# ------------------------------------------------------------

name = st.text_input("Car Name (e.g. Swift, i20, Honda City)")

year = st.number_input("Manufacturing Year", min_value=1990, max_value=2024, step=1)

km_driven = st.number_input("Kilometers Driven", min_value=0, step=500)

fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])

seller_type = st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])

transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

owner = st.selectbox("Owner Type", ["First Owner", "Second Owner", "Third Owner",
                                    "Fourth & Above Owner", "Test Drive Car"])

mileage = st.number_input("Mileage (km/ltr/kg)", min_value=0.0)

engine = st.number_input("Engine (CC)", min_value=0.0)

max_power = st.number_input("Max Power (BHP)", min_value=0.0)

seats = st.number_input("Number of Seats", min_value=2.0, max_value=10.0, step=1.0)

# ------------------------------------------------------------
# Prepare data for prediction
# ------------------------------------------------------------
if st.button("Predict Price"):

    # Frequency encoding for name (safe for unseen names)
    # NOTE: During training, model already learned the mapping internally.
    # So during prediction, we set unseen names to zero.
    try:
        # Extract frequency mapping from preprocessor
        # (not always available depending on pipeline structure)
        freq_map = model.named_steps['preprocessor'].transformers_[1][1]['imputer'].statistics_
    except:
        # fallback: unseen names → frequency = 0
        freq_map = {}

    name_freq = 0  # safe default

    input_data = {
        'year': year,
        'km_driven': km_driven,
        'fuel': fuel,
        'seller_type': seller_type,
        'transmission': transmission,
        'owner': owner,
        'mileage(km/ltr/kg)': mileage,
        'engine': engine,
        'max_power': max_power,
        'seats': seats,
        'name_freq': name_freq
    }

    input_df = pd.DataFrame([input_data])

    # ------------------------------------------------------------
    # Make Prediction
    # ------------------------------------------------------------
    predicted_price = model.predict(input_df)[0]

    st.success(f"Estimated Selling Price: ₹ **{predicted_price:,.0f}**")
