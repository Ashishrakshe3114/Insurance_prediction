import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open("best_model (1).pkl", "rb") as file:
    model = pickle.load(file)

st.title("💰 Insurance Charges Prediction App")
st.write("Predict medical insurance charges based on your personal details.")

# Sidebar for user input
st.sidebar.header("Enter your details:")

age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.sidebar.number_input("Number of Children", min_value=0, max_value=10, value=0)
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
region = st.sidebar.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# Create input DataFrame
input_dict = {
    "age": [age],
    "sex": [sex],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker],
    "region": [region]
}
input_df = pd.DataFrame(input_dict)

# Match model's training encoding (like in your notebook)
input_encoded = pd.get_dummies(input_df, drop_first=True)

# Ensure all expected columns exist (fill missing ones with 0)
model_features = model.feature_names_in_  # available in scikit-learn >=1.0
for col in model_features:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Arrange columns in the same order as training
input_encoded = input_encoded[model_features]

# Predict button
if st.button("Predict Charges"):
    prediction = model.predict(input_encoded)
    st.success(f"Estimated Insurance Charge: **${prediction[0]:,.2f}**")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Linear Regression Model")
