import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl

# Load the trained model and mappings
with open('model_stroke.pkl', 'rb') as f:
    model = pkl.load(f)

with open('label_mappings.pkl', 'rb') as f:
    mappings = pkl.load(f)

# Title and Description
st.title("Stroke Prediction App")
st.write("""
This app predicts the likelihood of a stroke based on various health and demographic factors. 
Provide the inputs below to get the prediction.
""")

# User input for prediction
st.sidebar.header("Input Features")

def user_input_features():
    gender = st.sidebar.selectbox("Gender", list(mappings['gender'].keys()))
    age = st.sidebar.slider("Age", 0, 100, 50)
    hypertension = st.sidebar.selectbox("Hypertension", ['No', 'Yes'])  # 0 for No, 1 for Yes
    heart_disease = st.sidebar.selectbox("Heart Disease", ['No', 'Yes'])  # 0 for No, 1 for Yes
    ever_married = st.sidebar.selectbox("Ever Married", list(mappings['ever_married'].keys()))
    work_type = st.sidebar.selectbox("Work Type", list(mappings['work_type'].keys()))
    residence_type = st.sidebar.selectbox("Residence Type", list(mappings['Residence_type'].keys()))
    avg_glucose_level = st.sidebar.slider("Average Glucose Level", 50, 300, 100)
    bmi = st.sidebar.slider("BMI", 10, 50, 25)
    smoking_status = st.sidebar.selectbox("Smoking Status", list(mappings['smoking_status'].keys()))
    
    # Encode categorical data
    data = {
        'gender': mappings['gender'][gender],
        'age': age,
        'hypertension': 1 if hypertension == 'Yes' else 0,
        'heart_disease': 1 if heart_disease == 'Yes' else 0,
        'ever_married': mappings['ever_married'][ever_married],
        'work_type': mappings['work_type'][work_type],
        'Residence_type': mappings['Residence_type'][residence_type],
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': mappings['smoking_status'][smoking_status],
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# Collect input features into a DataFrame
input_df = user_input_features()

# Display user inputs
st.subheader("User Input Features")
st.write(input_df)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    st.subheader("Prediction Result")
    if prediction[0] == 0:
        st.write("No Stroke Risk Detected")
    else:
        st.write("High Risk of Stroke Detected")
    
    st.subheader("Prediction Probability")
    st.write(f"Probability of No Stroke: {prediction_proba[0][0]:.2f}")
    st.write(f"Probability of Stroke: {prediction_proba[0][1]:.2f}")