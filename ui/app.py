import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl

from pathlib import Path
from ucimlrepo import fetch_ucirepo

@st.cache_data
def load_uci_dataset():
    heart_disease = fetch_ucirepo(id=45)

    X = np.array(heart_disease.data.features)
    y = np.array(heart_disease.data.targets)

    df = pd.DataFrame(np.c_[X, y], columns=heart_disease.data.headers)
    df.ffill(inplace=True)
    df.rename(columns={'num':'target'}, inplace=True)
    return df 

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
    
# LaTeX text formatter
def latex(label, expr="large"):
    return rf"$\textsf{{\{expr} {label}}}$"

def handle_input():
    # Get mapping for selectbox options
    mapping = display_name_id_mapping()
    
    # Input fields
    age = st.number_input(latex("Age"), min_value=20, max_value=100, value=50)
    sex = st.radio(latex("Sex"),  options=mapping['sex'], format_func=lambda record: record["display_name"])["id"]
    cp = st.selectbox(latex("Chest Pain Type"), options=mapping['cp'], format_func=lambda record: record["display_name"])["id"]
    trestbps = st.number_input(latex("Resting Blood Pressure"), min_value=80, max_value=200, value=120)
    chol = st.number_input(latex("Serum Cholesterol (mg/dl)"), min_value=100, max_value=600, value=200)
    fbs = st.radio(latex("Fasting Blood Sugar > 120 mg/dl"),  options=mapping['fbs'], format_func=lambda record: record["display_name"])["id"]
    restecg = st.selectbox(latex("Resting ECG Results"),  options=mapping['restecg'], format_func=lambda record: record["display_name"])["id"]
    thalach = st.number_input(latex("Max Heart Rate Achieved"), min_value=60, max_value=220, value=150)
    exang = st.radio(latex("Exercise Induced Angina"),  options=mapping['exang'], format_func=lambda record: record["display_name"])["id"]
    oldpeak = st.number_input(latex("ST Depression"), min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox(latex("Slope of ST Segment"),  options=mapping['slope'], format_func=lambda record: record["display_name"])["id"]
    ca = st.selectbox(latex("Major Vessels (0-3) Colored by Flourosopy"), [0, 1, 2, 3])
    thal = st.selectbox(latex("Thalassemia"),  options=mapping['thal'], format_func=lambda record: record["display_name"])["id"]

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
    # Get path relative to this script's folder
    model_path = Path(__file__).resolve().parent.parent / "models" / "final_model.pkl"
    with open(model_path, "rb") as f:
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

@st.cache_data
def create_charts(df):
    # Plot correlation heatmap
    st.write(latex("Correlation Heatmap:"))
    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(plt)

    # Distribution of target
    st.write(latex("Heart Disease Distribution:"))
    plt.figure(figsize=(10,6))
    sns.countplot(x='target', data=df, hue='target', palette='rocket_r', legend=False)
    st.pyplot(plt)


def handle_data_visualization():
    uploaded_file = st.file_uploader(latex("Upload dataset (CSV) to explore trends (the default is the UCI heart disease dataset)"), type=["csv"])

    if uploaded_file is None:
        df = load_uci_dataset()
    else:
        df = pd.read_csv(uploaded_file)

    st.write(latex("Dataset Preview:"))
    st.dataframe(df)

    create_charts(df)


            
def main():
    st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
    st.title("❤️ Heart Disease: Prediction & Exploration")
    st.subheader("This application predicts the presence and severity of heart disease based on user-provided health metrics.")
    model = load_model()
    input_data = handle_input()
    handle_prediction(model, input_data)
    st.subheader("📊 Heart Disease Dataset Trends")
    handle_data_visualization()

if __name__ == "__main__":
    main()