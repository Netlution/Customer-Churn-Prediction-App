import streamlit as st
import pandas as pd
import joblib  

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("Customer Churn Prediction (Classification) 🅰️")

st.write(
    "This application predicts customer churn — "
    "whether a customer will leave (1) or stay (0) — "
    "based on their usage patterns, subscription details, and demographic factors."
)

# Sidebar inputs
st.sidebar.header("Classification Input Features")

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
customer_value = st.sidebar.number_input("Customer Value", min_value=0, max_value=1000, value=50, step=1)

# Collect Data
input_data = {
    "Call Failure": call_failure,
    "Complains": complains,
    "Subscription Length": subscription_length,
    "Charge Amount": charge_amount,
    "Seconds of Use": seconds_of_use,
    "Frequency of use": frequency_of_use,
    "Frequency of SMS": frequency_of_sms,
    "Distinct Called Numbers": distinct_called_numbers,
    "Age Group": age,
    "Tariff Plan": tariff_plan,
    "Status": status,
    "Customer Value": customer_value
}

df_input = pd.DataFrame([input_data])

st.subheader("📋 User Input Summary")
st.dataframe(df_input)

if st.button("Predict"):
    model = joblib.load("rf_churn_model.pkl")
    prediction = model.predict(df_input)
    
    if prediction[0] == 1:
        st.error("The model predicts: Customer will leave ❌")
    else:
        st.success("The model predicts: Customer will stay ✅")

# Move project description here (below summary and prediction)
st.markdown("""
### 📖 Customer Churn Prediction (Classification)

Customer churn is a major challenge in telecom and subscription-based businesses.  
This module applies a **classification model** to predict whether a customer will churn  
(**1 = leave**) or remain (**0 = stay**).

**Features considered include:**
- Call Failures & Complaints  
- Subscription Length  
- Charge Amount & Usage Patterns  
- Frequency of Use (Calls & SMS)  
- Distinct Called Numbers  
- Age Group  
- Tariff Plan & Status  
- Customer Value  

With these inputs, the model predicts whether the customer is at risk of leaving.

✅ **Output Options:**  
- ❌ Customer will leave (high churn risk)  
- ✅ Customer will stay (low churn risk)  
""")

st.image("././static/Classification model.png", caption="Classification model")



