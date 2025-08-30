import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
confusion_matrix, RocCurveDisplay
)

# Caching for display name to id mapping
@st.cache_data
def display_name_id_mapping():
    mapping = {}
    entry = lambda id, display_name: {"id": id, "display_name": display_name}

    mapping['sex'] = [
        entry(1, 'Male'),
        entry(0, 'Female')
    ]
    mapping['cp'] = [
        entry(1, 'Typical Angina'),
        entry(2, 'Atypical Angina'),
        entry(3, 'Non-anginal pain'),
        entry(4, 'Asymptomatic')
    ]
    mapping['fbs'] = [
        entry(0, 'False'),
        entry(1, 'True')
    ]
    mapping['restecg'] = [
        entry(0, 'Normal'),
        entry(1, 'Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)'),
        entry(2, 'Showing probable or definite left ventricular hypertrophy by Estes\' criteria')
    ]
    mapping['exang'] = [
        entry(0, 'No'),
        entry(1, 'Yes')
    ]
    mapping['slope'] = [
        entry(1, 'Upsloping'),
        entry(2, 'Flat'),
        entry(3, 'Downsloping')
    ]
    mapping['thal'] = [
        entry(3, 'Normal'),
        entry(6, 'Fixed Defect'),
        entry(7, 'Reversible Defect')
    ]
    return mapping
    


def handle_input():
    # Get mapping for selectbox options
    mapping = display_name_id_mapping()
    # LaTeX label formatter
    label_latex = lambda label, expr="large": rf"$\textsf{{\{expr} {label}}}$"

    # Input fields
    age = st.number_input(label_latex("Age"), min_value=20, max_value=100, value=50)
    sex = st.radio(label_latex("Sex"),  options=mapping['sex'], format_func=lambda record: record["display_name"])["id"]
    cp = st.selectbox(label_latex("Chest Pain Type"), options=mapping['cp'], format_func=lambda record: record["display_name"])["id"]
    trestbps = st.number_input(label_latex("Resting Blood Pressure"), min_value=80, max_value=200, value=120)
    chol = st.number_input(label_latex("Serum Cholesterol (mg/dl)"), min_value=100, max_value=600, value=200)
    fbs = st.radio(label_latex("Fasting Blood Sugar > 120 mg/dl"),  options=mapping['fbs'], format_func=lambda record: record["display_name"])["id"]
    restecg = st.selectbox(label_latex("Resting ECG Results"),  options=mapping['restecg'], format_func=lambda record: record["display_name"])["id"]
    thalach = st.number_input(label_latex("Max Heart Rate Achieved"), min_value=60, max_value=220, value=150)
    exang = st.radio(label_latex("Exercise Induced Angina"),  options=mapping['exang'], format_func=lambda record: record["display_name"])["id"]
    oldpeak = st.number_input(label_latex("ST Depression"), min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox(label_latex("Slope of ST Segment"),  options=mapping['slope'], format_func=lambda record: record["display_name"])["id"]
    ca = st.selectbox(label_latex("Major Vessels (0-3) Colored by Flourosopy"), [0, 1, 2, 3])
    thal = st.selectbox(label_latex("Thalassemia"),  options=mapping['thal'], format_func=lambda record: record["display_name"])["id"]

    # Collect input in DataFrame
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })

    return input_data

# Load model with caching
@st.cache_resource
def load_model():
    with open("../models/final_model.pkl", "rb") as f:
        model = pkl.load(f)
        return model
    
def handle_prediction(model, input_data):
    # Predict when button is clicked
    if st.button("Predict", width=200):
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][int(prediction)]

        if prediction == 0:
            st.success(f"The model predicts that you do NOT have heart disease. (Probability: {prob:.0%})")
        elif prediction == 1:
            st.warning(f"The model predicts that you HAVE MILD heart disease. (Probability: {prob:.0%})")
        elif prediction == 2:
            st.warning(f"The model predicts that you HAVE MODERATE heart disease. (Probability: {prob:.0%})")
        elif prediction == 3:
            st.error(f"The model predicts that you HAVE SEVERE heart disease. (Probability: {prob:.0%})")
        elif prediction == 4:
            st.error(f"The model predicts that you HAVE CRITICAL heart disease. (Probability: {prob:.0%})")
        else:
            st.error("Unexpected prediction value.")
            
def main():
    st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
    st.title("❤️ Heart Disease: Prediction & Exploration")
    st.subheader("This application predicts the presence and severity of heart disease based on user-provided health metrics.")
    model = load_model()
    input_data = handle_input()
    handle_prediction(model, input_data)

if __name__ == "__main__":
    main()