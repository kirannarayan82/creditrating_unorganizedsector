import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit app title
st.title('Credit Risk Prediction for Unorganized Sector')

# Helper function to get user input
def get_user_input():
    age = st.number_input('Age', min_value=18, max_value=65, value=25)
    employment_years = st.number_input('Employment Years', min_value=1, max_value=40, value=5)
    income = st.number_input('Monthly Income', min_value=5000, max_value=50000, value=15000)
    debt = st.number_input('Total Debt', min_value=0, max_value=25000, value=5000)
    marital_status = st.selectbox('Marital Status', ['Single', 'Married'])
    education_level = st.selectbox('Education Level', ['Non-Graduate', 'Graduate'])
    utility_payment_history = st.selectbox('Utility Payment History', ['Irregular/Non-Payer', 'Regular Payer'])
    mobile_phone_usage = st.number_input('Monthly Mobile Phone Usage (in INR)', min_value=500, max_value=5000, value=1500)
    
    # Convert categorical inputs to numerical
    marital_status = 1 if marital_status == 'Married' else 0
    education_level = 1 if education_level == 'Graduate' else 0
    utility_payment_history = 1 if utility_payment_history == 'Regular Payer' else 0
    
    # Create a DataFrame for input
    input_data = pd.DataFrame({
        'Age': [age],
        'Employment_Years': [employment_years],
        'Income': [income],
        'Debt': [debt],
        'Marital_Status': [marital_status],
        'Education_Level': [education_level],
        'Utility_Payment_History': [utility_payment_history],
        'Mobile_Phone_Usage': [mobile_phone_usage]
    })
    
    return input_data

# Get user input
user_input = get_user_input()

# Standardize the input data
user_input_scaled = scaler.transform(user_input)

# Make prediction
prediction = model.predict(user_input_scaled)
prediction_proba = model.predict_proba(user_input_scaled)

# Display the result
st.write(f'Creditworthiness Prediction: {"Good" if prediction[0] == 1 else "Bad"}')
st.write(f'Probability of Good Credit: {prediction_proba[0][1]:.2f}')
st.write(f'Probability of Bad Credit: {prediction_proba[0][0]:.2f}')
