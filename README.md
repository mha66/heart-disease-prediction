# ❤️ Heart Disease Prediction App

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)](https://scikit-learn.org/)

## 📌 Project Overview
This project predicts **heart disease levels (0–4)** based on patient medical data from the [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease).  
It includes data preprocessing, dimensionality reduction, feature selection, supervised and unsupervised learning, and a **Streamlit web app** for real-time predictions.

---

## 🎯 Goals
1. Predict severity of heart disease using ML.  
2. Provide an interactive **Streamlit UI** for predictions.  
3. Allow users to **explore dataset trends** via visualizations.  
4. Support both **local deployment** and **Ngrok sharing**.

---

## 🤖 Live Demo

You can try the app here: [Heart Disease Prediction App](https://likely-holy-tapir.ngrok-free.app)  
*(Note: Ngrok links are temporary and may expire. Run locally if the link is unavailable.)*

---

## 📂 Project Structure
```
HEART_DISEASE_PROJECT
│── data/                         
│   └── heart_disease.csv
│
│── deployment/                   
│   ├── ngrok_setup.txt
│   └── run_app.bat
│
│── models/                       
│   ├── decision_tree_model.pkl
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── svm_model.pkl
│   └── final_model.pkl           # Best model
│
│── notebooks/                    
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_pca_analysis.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_supervised_learning.ipynb
│   ├── 05_unsupervised_learning.ipynb
│   └── 06_hyperparameter_tuning.ipynb
│
│── results/                      
│   ├── evaluation_metrics.txt
│   └── hyperparameter_tuning_metrics.csv
│
│── ui/                           
│   └── app.py
│
│── requirements.txt              
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/mha66/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate     # On Windows
source venv/bin/activate  # On Mac/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Streamlit App

### Local Deployment
```bash
streamlit run ui/app.py
```
Access it at: [http://localhost:8501](http://localhost:8501)

### Public Deployment (Ngrok)
```bash
streamlit run ui/app.py
ngrok http 8501
```
Copy the generated **public URL** and share it.

*(Note: check ngrok_setup.txt for more details about public deployment)*

---

## 📊 Model Training & Results
- **Dataset:** UCI Heart Disease (303 samples, 14 features → 19 after preprocessing).  
- **Dimensionality Reduction:** PCA → 11 components.  
- **Feature Selection:** RFE → 7 top features.  
- **Supervised Models Tried:** Logistic Regression, Decision Tree, Random Forest, SVM.
- **Unsupervised Models Tried:** K-Means Clustering, Hierarchical Clustering  
- **Best Model:** Random Forest Classifier (Randomized Search).  
- **Accuracy:** ~69% (multi-class classification).
- **ROC-AUC Score:** ~0.84 (One-vs-Rest Strategy)

Results are stored in [`results/`](./results).

---

## 🖼️ App Preview
<p align="center">
<img width="2428" height="1277" alt="Screenshot 2025-09-02 015847" src="https://github.com/user-attachments/assets/280abe4d-0168-404a-bed5-f3e06e50a597" />
<img width="2429" height="1103" alt="Screenshot 2025-09-02 015943" src="https://github.com/user-attachments/assets/65b3ec59-df20-4804-bbd0-3c13adef522e" />
<img width="2467" height="1101" alt="Screenshot 2025-09-02 020035" src="https://github.com/user-attachments/assets/39e4020d-7766-46bc-9008-cb3e18a9642c" />
<img width="2468" height="950" alt="Screenshot 2025-09-02 020122" src="https://github.com/user-attachments/assets/7d1f4a0e-11d6-47e5-865f-a9576331962c" />
</p>

---

## 📜 License
This project is licensed under the MIT License.  
