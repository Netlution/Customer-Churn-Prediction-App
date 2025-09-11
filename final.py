# import streamlit as st
# import pandas as pd
# import joblib  

# st.set_page_config(page_title="Customer Churn Prediction App", layout="wide")

# st.title("Customer Churn Prediction App")
# st.write("This application leverages machine learning to predict customer churn whether a customer will leave or  stay **1 or 0** their service based on their usage patterns, subscription details, and demographic factors.")


# # Sidebar Inputsst.sidebar.header("User Input Features")
# age = st.sidebar.number_input("Age Group", min_value=17, max_value=90, value=38, step=1)

# call_failure = st.sidebar.selectbox("Call Failure", options=[
#     8, 0, 10, 3, 11, 4, 13, 7, 6, 9, 25, 2, 23, 21, 1, 16, 12,
#     14, 28, 5, 26, 24, 19, 15, 22, 20, 18, 17, 30, 27, 29, 31,
#     33, 35, 32, 34, 36
# ])

# complains = st.sidebar.selectbox("Complains", options=[
#        0, 1])

# subscription_length = st.sidebar.selectbox("Subscription Length", options=[
#     38, 39, 37, 33, 36, 34, 35, 31, 27, 26, 25, 18, 17, 15, 16, 9, 40,
#     41, 29, 28, 20, 19, 11, 32, 24, 23, 13, 14, 7, 42, 43, 30, 22, 21,
#     12, 5, 44, 45, 10, 3, 6, 8, 4, 46, 47
# ])


# charge_amount = st.sidebar.selectbox("Charge Amount", options=[
#     0, 1, 2, 3, 8, 4, 9, 7, 5, 10, 6
# ])


# frequency_of_sms = st.sidebar.selectbox("Frequency of SMS", options=[
#     5, 7, 359, 1, 2, 32, 285, 144, 0, 8, 54, 483, 150, 186, 13, 384, 11, 108, 16, 26, 
#     9, 34, 14, 4, 271, 38, 175, 193, 30, 10, 21, 19, 28, 85, 73, 31, 215, 12, 364, 6, 
#     37, 290, 149, 59, 488, 155, 191, 18, 389, 113, 39, 276, 43, 180, 198, 35, 15, 24, 
#     33, 90, 78, 36, 220, 354, 27, 280, 139, 3, 49, 478, 145, 181, 379, 103, 29, 266, 
#     170, 188, 25, 23, 80, 68, 210, 17, 369, 42, 295, 154, 64, 493, 160, 196, 394, 118, 
#     44, 281, 48, 185, 203, 40, 20, 95, 83, 41, 225, 349, 22, 275, 134, 473, 140, 176, 
#     374, 98, 261, 165, 183, 75, 63, 205, 47, 300, 159, 69, 498, 201, 399, 123, 286, 53, 
#     190, 208, 45, 100, 88, 46, 230, 344, 270, 129, 468, 135, 171, 93, 256, 178, 70, 58, 
#     200, 370, 296, 65, 494, 161, 197, 395, 119, 282, 204, 96, 84, 226, 348, 274, 133, 
#     472, 373, 97, 260, 164, 182, 74, 62, 380, 306, 504, 207, 405, 55, 292, 214, 51, 106, 
#     94, 52, 236, 338, 264, 462, 363, 87, 250, 172, 194, 391, 317, 86, 515, 218, 416, 66, 
#     303, 60, 117, 105, 247, 327, 253, 112, 451, 352, 76, 239, 143, 368, 294, 153, 492, 
#     195, 393, 184, 202, 82, 224, 350, 474, 141, 177, 375, 99, 262, 166, 206, 351, 277, 
#     136, 475, 142, 376, 263, 167, 77, 367, 293, 152, 491, 158, 392, 116, 279, 81, 223, 
#     334, 458, 125, 246, 168, 278, 137, 476, 179, 377, 101, 357, 283, 481, 148, 382, 269, 
#     173, 71, 213, 347, 273, 132, 471, 138, 174, 372, 259, 163, 61, 362, 288, 147, 57, 
#     486, 189, 387, 111, 342, 268, 127, 466, 169, 91, 254, 56, 337, 122, 461, 128, 249, 
#     289, 487, 388, 89, 219, 341, 267, 126, 465, 366, 157, 67, 299, 497, 398, 229, 331, 
#     257, 455, 356, 243, 187, 310, 79, 508, 211, 409, 110, 240, 320, 444, 345, 232, 361, 
#     287, 146, 485, 386, 217, 343, 467, 92, 255, 199, 360, 484, 151, 385, 109, 272, 216, 
#     490, 115, 222, 371, 297, 156, 495, 162, 396, 120, 50, 227, 302, 500, 401, 192, 102, 
#     480, 381, 212, 307, 505, 406, 130, 107, 237, 72, 501, 402, 233, 355, 479, 104, 313, 
#     511, 412, 221, 469, 324, 522, 423, 124, 301, 499, 400, 209, 231, 358, 284, 482, 383
# ])


# status = st.sidebar.selectbox("Status", options=[1, 2])

# call_failure = st.sidebar.number_input("Call Failure", min_value=0, max_value=100000, value=0, step=100)
# complains = st.sidebar.number_input("Complains", min_value=0, max_value=100000, value=0, step=100)
# charge_amount = st.sidebar.number_input("Charge  Amount", min_value=0, max_value=20000, value=40, step=1)
# # Collect Data

# input_data = {
#     "age": age_group,
#     "call_failure": call_failure,
#     "complain": complains,
#     "subscription_length": subscription_length,
#     "charge_amount": charge_amount,
#     "seconds_of_use": seconds_of_use,
#     "frequency_of_use": frequency_of_use,
#     "frequency_of_sms": frequency_of_sms,
#     "distinct_called_numbers": distinct_called_numbers,
#     "tariff_plan": tariff_plan,
#     "status": status,
#     "customer_value": customer_value
# }


# df_input = pd.DataFrame([input_data])

# st.subheader("User Input Summary")
# st.dataframe(df_input)



# if st.button("Predict"):
#     # Model Prediction (replace with your model)
#     model = joblib.load ("churn_model.pkl")
#     prediction = model.predict(df_input)    
#     print(prediction)
#     if prediction[0] == 1:
#         st.success("a customer will leave")
#     elif:
#       if prediction[0] == 0:
#           st.success ("a customer will stay")
#     else:
#         st.success("not a customer")


import streamlit as st
import pandas as pd
import joblib  

st.set_page_config(page_title="Customer Churn Prediction App", layout="wide")

st.title("Customer Churn Prediction App")
st.write("This application leverages machine learning to predict customer churn — whether a customer will leave (1) or stay (0) — based on their usage patterns, subscription details, and demographic factors.")

# Sidebar Inputs
st.sidebar.header("User Input Features")

call_failure = st.sidebar.number_input("Call Failure", min_value=0, max_value=100, value=0, step=1)
complains = st.sidebar.selectbox("Complains", options=[0, 1])
subscription_length = st.sidebar.number_input("Subscription Length (months)", min_value=1, max_value=60, value=12, step=1)
charge_amount = st.sidebar.number_input("Charge Amount", min_value=0, max_value=20000, value=40, step=1)
seconds_of_use = st.sidebar.number_input("Seconds of Use", min_value=0, max_value=100000, value=300, step=100)
frequency_of_use = st.sidebar.number_input("Frequency of Use", min_value=0, max_value=1000, value=10, step=1)
frequency_of_sms = st.sidebar.number_input("Frequency of SMS", min_value=0, max_value=1000, value=5, step=1)
distinct_called_numbers = st.sidebar.number_input("Distinct Called Numbers", min_value=0, max_value=1000, value=20, step=1)
age = st.sidebar.number_input("Age Group", min_value=17, max_value=90, value=38, step=1)
tariff_plan = st.sidebar.selectbox("Tariff Plan", options=[0, 1])  # Example: 0=Basic, 1=Premium
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

st.subheader("User Input Summary")
st.dataframe(df_input)

# Prediction
if st.button("Predict"):
    model = joblib.load("churn_model.pkl")
    prediction = model.predict(df_input)    
    
    if prediction[0] == 1:
        st.success("The model predicts: Customer will leave ❌")
    else:
        st.success("The model predicts: Customer will stay ✅")
