import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open("best_model (1).pkl", "rb") as file:
    model = pickle.load(file)

# App title
st.title("💰 Insurance Charges Prediction App")
st.write("Predict medical insurance charges based on your personal details.")

# Sidebar inputs
st.sidebar.header("Enter your details:")

age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.sidebar.number_input("Number of Children", min_value=0, max_value=10, value=0)
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
region = st.sidebar.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# Encode categorical values as per model training
sex = 1 if sex == "male" else 0
smoker = 1 if smoker == "yes" else 0
region_map = {"southwest": 0, "southeast": 1, "northwest": 2, "northeast": 3}
region = region_map[region]

# Create DataFrame for model input
input_data = pd.DataFrame(
    [[age, sex, bmi, children, smoker, region]],
    columns=["age", "sex", "bmi", "children", "smoker", "region"]
)

# Predict button
if st.button("Predict Charges"):
    prediction = model.predict(input_data)
    st.success(f"Estimated Insurance Charge: **${prediction[0]:,.2f}**")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Linear Regression Model")
