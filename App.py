import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained model
model = joblib.load("house_price_model.pkl")

# Define feature names
feature_columns = ["Location", "Area", "Bedroom", "Bathroom", "Floor", "View", "Facing", "Elevator", "Payment System"]

# Mapping dictionaries
location_mapping = {
    "New Administrative Capital": 1, "First Settlement": 2, "Second Settlement": 3, "Third Settlement": 4,
    "Fourth Settlement": 5, "Fifth Settlement": 6, "Madinaty": 7, "El Rehab": 8, "Kattameya": 9,
    "Zahraa Nasr City": 10, "Nasr City": 11, "Maadi": 12, "Muqattam": 13, "Zahraa El Maadi": 14,
    "Sheraton": 15, "Heliopolis": 16, "Badr": 17, "El Obour": 18, "El Mostakbal": 19, "El Shorouk": 20,
    "El Salam": 21, "Gesr El Suez": 22, "El Marg": 23, "Ain Shams": 24, "Mostorod": 25, "Shubra El Kheima": 26,
    "Hadayek El Kobba": 27, "El Zawya El Hamra": 28, "Shubra": 29, "Rod El Farag": 30, "El Basatin": 31,
    "El Abbasiya": 32, "Sayeda Zeinab": 33, "Imbaba": 34, "El Azbakeya": 35, "El Fustat": 36, "Agouza": 37,
    "Boulak El Dakrour": 38, "El Mataria": 39, "El Nozha": 40, "El Waili": 41, "Abdeen": 42, "El Sharabiya": 43,
    "El Amireya": 44, "El Manial": 45, "Zamalek": 46, "El Mohandessin": 47, "Dokki": 48, "El Haram": 49,
    "El Warraq": 50, "El Omraniya": 51, "El Moneeb": 52, "Hadayek El Ahram": 53, "Sheikh Zayed": 54,
    "Hadayek October": 55, "New Sphinx": 56, "6th of October": 57, "El Faggala": 58, "Helwan": 59, "Faisal": 60
}

view_mapping = {"Garden": 1, "Street": 2, "Apartment": 3, "Apartment Building": 4}
facing_mapping = {"North": 1, "South": 2}
elevator_mapping = {"Yes": 1, "No": 0}
payment_mapping = {"Cash": 0, "Installments": 1}

# Streamlit Page Config
st.set_page_config(page_title="🏠 House Price Prediction", layout="centered")

st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #89f7fe, #66a6ff);
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# App Title
st.markdown('<h1>🏠 House Price Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p>Enter the details to predict house price in Egypt.</p>', unsafe_allow_html=True)

# Input Widgets
location = st.selectbox("📍 Location", list(location_mapping.keys()))
area = st.slider("📏 Area (sqm)", 90, 250, step=10)
bedroom = st.number_input("🛏 Bedroom", min_value=0, step=1)
bathroom = st.number_input("🚻 Bathroom", min_value=0, step=1)
floor = st.number_input("🏢 Floor", min_value=0, step=1)
view = st.radio("🌄 View", list(view_mapping.keys()))
facing = st.radio("🧭 Facing", list(facing_mapping.keys()))
elevator = st.radio("🚪 Elevator", ["Yes", "No"])
payment = st.radio("💰 Payment System", ["Cash", "Installments"])

# Prediction function
def predict_price(location, area, bedroom, bathroom, floor, view, facing, elevator, payment):
    location = location_mapping.get(location, 0)
    view = view_mapping.get(view, 0)
    facing = facing_mapping.get(facing, 0)
    elevator = elevator_mapping.get(elevator, 0)
    payment = payment_mapping.get(payment, 0)
    
    input_data = np.array([[location, area, bedroom, bathroom, floor, view, facing, elevator, payment]])
    input_df = pd.DataFrame(input_data, columns=feature_columns)
    
    predicted_price = model.predict(input_df)[0]
    return predicted_price

# Buttons
if st.button("Predict Price"):
    result = predict_price(location, area, bedroom, bathroom, floor, view, facing, elevator, payment)
    st.success(f"💰 Predicted House Price: {result:,.2f} EGP")

if st.button('Clear'):
    st.rerun()
