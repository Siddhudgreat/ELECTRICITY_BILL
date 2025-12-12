import streamlit as st
import pandas as pd
import pickle

# Load model & encoders
model = pickle.load(open("electricity_model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))
le_season = encoders["season"]

st.title("⚡ Electricity Bill Prediction App")

st.write("Enter details below to estimate your monthly electricity bill:")

# Inputs
family_members = st.slider("Number of Family Members", 1, 10)
daily_usage = st.slider("Daily Usage Hours (Avg)", 4.0, 16.0, step=0.1)
ac_usage = st.slider("AC Usage Hours (Avg)", 0.0, 12.0, step=0.1)
appliances = st.slider("Total Number of Appliances", 3, 25)
fridge_rating = st.selectbox("Refrigerator Efficiency (Star Rating)", [1, 2, 3, 4, 5])
unit_price = st.slider("Price Per Unit (₹)", 5.0, 10.0, step=0.1)
season = st.selectbox("Season", le_season.classes_)

# Prepare input
input_df = pd.DataFrame({
    "FamilyMembers": [family_members],
    "DailyUsageHours": [daily_usage],
    "ACUsageHours": [ac_usage],
    "ApplianceCount": [appliances],
    "FridgeEfficiency": [fridge_rating],
    "UnitPrice": [unit_price],
    "Season": [le_season.transform([season])[0]],
    "UnitsConsumed": [0]  # placeholder, model expects column
})

# Predict
if st.button("Predict Bill"):
    prediction = model.predict(input_df)[0]
    st.success(f"🧾 Estimated Monthly Electricity Bill: ₹ {round(prediction, 2)}")
