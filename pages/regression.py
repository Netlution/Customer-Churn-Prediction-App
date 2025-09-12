import streamlit as st
import pandas as pd
import joblib  

st.title("📈 Customer Value Regression (Continuous Prediction)")
st.write("This module predicts the **continuous customer value** using regression models.") 

# Sidebar inputs
st.sidebar.header("Regression Input Features")

call_failure = st.sidebar.number_input("Call Failure", min_value=0, max_value=100, value=0, step=1)
complains = st.sidebar.selectbox("Complains", options=[0, 1])
subscription_length = st.sidebar.number_input("Subscription Length (months)", min_value=1, max_value=60, value=12, step=1)
charge_amount = st.sidebar.number_input("Charge Amount", min_value=0, max_value=20000, value=40, step=1)
seconds_of_use = st.sidebar.number_input("Seconds of Use", min_value=0, max_value=100000, value=300, step=100)
frequency_of_use = st.sidebar.number_input("Frequency of Use", min_value=0, max_value=1000, value=10, step=1)
frequency_of_sms = st.sidebar.number_input("Frequency of SMS", min_value=0, max_value=1000, value=5, step=1)
distinct_called_numbers = st.sidebar.number_input("Distinct Called Numbers", min_value=0, max_value=1000, value=20, step=1)
age = st.sidebar.number_input("Age Group", min_value=17, max_value=90, value=38, step=1)
tariff_plan = st.sidebar.selectbox("Tariff Plan", options=[0, 1])  
status = st.sidebar.selectbox("Status", options=[1, 2])

# Collect input data
input_data = {
    "Call Failure": call_failure,
    "Complains": complains,
    "Subscription Length": subscription_length,
    "Charge Amount": charge_amount,
    "Seconds of Use": seconds_of_use,
    "Frequency of Use": frequency_of_use,   # ✅ fixed capitalization
    "Frequency of SMS": frequency_of_sms,
    "Distinct Called Numbers": distinct_called_numbers,
    "Age Group": age,
    "Tariff Plan": tariff_plan,
    "Status": status
}

df_input = pd.DataFrame([input_data])

# Show user input
st.subheader("📋 User Input Summary")
st.dataframe(df_input)

# Cache model load
@st.cache_resource
def load_model():
    return joblib.load("rf_regression_model.pkl")

# Prediction button
if st.button("Predict"):
    model = load_model()
    prediction = model.predict(df_input)    
    st.success(f"The model predicts: Estimated Value = **{prediction[0]:.2f}**")

