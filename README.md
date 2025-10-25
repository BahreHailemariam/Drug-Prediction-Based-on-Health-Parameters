# 💊 Drug Prediction Based on Health Parameters

## 📘 Overview
This project predicts the most suitable **drug type or treatment** for a patient based on their health parameters such as **age, sex, blood pressure, cholesterol, and sodium-to-potassium ratio**. Using machine learning algorithms, the model assists healthcare professionals in prescribing appropriate medications efficiently and accurately.

## 🎯 Business Objective
To build a **predictive system** that helps healthcare organizations and practitioners:
- Recommend optimal drug categories based on patient data  
- Reduce trial-and-error in prescriptions  
- Improve patient satisfaction and treatment effectiveness  

## 🧩 Workflow

### 1️⃣ Define the Problem
Identify the goal: predict which **drug type** a patient should be prescribed based on physiological parameters.

### 2️⃣ Data Extraction & Cleaning
- Load dataset (`drug200.csv` or healthcare API)
- Remove missing or duplicate records
- Standardize categorical values (`Male/Female`, `High/Normal/Low`)

```python
import pandas as pd
df = pd.read_csv("drug200.csv")
df.drop_duplicates(inplace=True)
df.fillna(method='ffill', inplace=True)
```
### 3️⃣ Data Preprocessing
Encode categorical variables and scale numerical features.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['BP'] = le.fit_transform(df['BP'])
df['Cholesterol'] = le.fit_transform(df['Cholesterol'])

scaler = StandardScaler()
df[['Age','Na_to_K']] = scaler.fit_transform(df[['Age','Na_to_K']])

X = df.drop('Drug', axis=1)
y = df['Drug']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
### 4️⃣ Model Training
Train multiple models (Decision Tree, Random Forest, Logistic Regression) and evaluate.

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```
### 5️⃣ Deployment (Streamlit)

```python
import streamlit as st
import numpy as np

st.title("💊 Drug Prediction App")
age = st.number_input("Age", 1, 100)
sex = st.selectbox("Sex", ["Male", "Female"])
bp = st.selectbox("Blood Pressure", ["Low", "Normal", "High"])
chol = st.selectbox("Cholesterol", ["Normal", "High"])
na_to_k = st.number_input("Sodium to Potassium Ratio", 0.0, 50.0)

if st.button("Predict Drug"):
    input_data = pd.DataFrame([[age, sex, bp, chol, na_to_k]],
                              columns=['Age','Sex','BP','Cholesterol','Na_to_K'])
    prediction = model.predict(input_data)
    st.success(f"Recommended Drug: {prediction[0]}")
```
# Power BI Report Specification  
### Project: Drug Prediction Based on the Health Parameters of a Patient  

---

## 🎯 Purpose  
This Power BI dashboard helps visualize patient health data, identify key risk indicators, and monitor drug prediction outcomes from a machine learning model.  
It provides both clinical insights (health parameter trends, drug prediction patterns) and operational metrics (prediction accuracy, patient distribution).

---

## 📦 Data Sources  
| Source | Description | Format |
|---------|--------------|--------|
| `patient_data.csv` | Raw patient health data (Age, BP, Cholesterol, etc.) | CSV |
| `model_predictions.csv` | Output from ML model with predicted drug type | CSV |
| `hospital_info.sql` | Optional database of hospital & department details | SQL |

---

## 🧱 Data Model  
**Fact Table:** `Fact_Predictions`  
**Dimension Tables:**
- `Dim_Patient`
- `Dim_Drug`
- `Dim_HealthMetrics`
- `Dim_Time`

### Relationships  
- `Dim_Patient[PatientID]` → `Fact_Predictions[PatientID]`  
- `Dim_Drug[DrugID]` → `Fact_Predictions[PredictedDrugID]`  
- `Dim_Time[DateKey]` → `Fact_Predictions[PredictionDateKey]`

---

## ⚙️ Data Transformation Steps  
1. Load all CSV/SQL data using Power Query.  
2. Remove duplicates and handle missing values (replace or drop).  
3. Normalize categorical values (e.g., Gender, BP level).  
4. Convert dates to a proper `DateKey` format.  
5. Merge datasets to create a unified `Fact_Predictions` table.  
6. Add calculated columns (BMI, Risk Index, Prediction Accuracy).  
7. Create measure tables for aggregations.

---

## 📊 Dashboard Pages and Visuals  

### **1. Patient Overview**
- **KPIs**: Total Patients, Avg Age, Gender Distribution, Avg BMI  
- **Visuals**:  
  - Pie Chart: Gender split  
  - Histogram: Age distribution  
  - Gauge: Average BMI  
  - Card: High-risk patient count  

### **2. Health Parameter Insights**
- **KPIs**: Avg BP, Avg Cholesterol, Avg Na-to-K ratio  
- **Visuals**:  
  - Line Chart: Blood Pressure vs Age  
  - Heatmap: Cholesterol vs Drug Type  
  - Bar Chart: Na-to-K ratio per Drug  

### **3. Drug Prediction Results**
- **KPIs**: Prediction Accuracy %, Top Predicted Drugs  
- **Visuals**:  
  - Confusion Matrix (using Python visual)  
  - Column Chart: Drug predictions by age group  
  - Card visuals: Precision, Recall, F1-Score  

### **4. Model Performance**
- **Visuals**:  
  - Line Chart: Accuracy trend over time  
  - Scatter Plot: True vs Predicted probabilities  
  - KPI Cards: Accuracy %, AUC, Recall, F1  

### **5. Operational & Regional Insights (Optional)**
- **Visuals**:  
  - Map: Patient distribution by city  
  - Table: Top performing clinics  
  - Line Chart: Drug usage trend by month  

---

## 📏 Key DAX Measures  

```DAX
Total Patients = COUNTROWS('Dim_Patient')

Avg BMI = AVERAGE('Dim_HealthMetrics'[BMI])

Prediction Accuracy % = 
DIVIDE(
    SUM('Fact_Predictions'[CorrectPredictions]),
    COUNTROWS('Fact_Predictions'),
    0
)

High Risk Patients = 
CALCULATE(
    COUNTROWS('Dim_Patient'),
    FILTER('Dim_HealthMetrics', 'Dim_HealthMetrics'[RiskIndex] > 0.8)
)
```

## 📊 Dashboard Insights (Power BI)
- **Drug Distribution by Age and Gender**: Reveals demographic patterns.
- **BP vs Cholesterol Impact**: Displays how blood pressure and cholesterol influence drug recommendations.
- **Na/K Ratio Influence**: Identifies key features driving drug prediction.
- **Model Performance Metrics**: Shows confusion matrix, accuracy, precision, and recall.

## 🛠️ Tech Stack
Python | Pandas | Scikit-learn | Matplotlib | Streamlit | Power BI | Airflow

## 📂 Project Structure
```
drug-prediction/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── Model_Training.ipynb
│
├── src/
│   ├── load_data.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── app.py
│
├── models/
│   ├── drug_model.pkl
│
├── reports/
│   ├── PowerBI_Dashboard.pbix
│   ├── metrics_summary.csv
│
├── requirements.txt
└── README.md
```
## 👤 Author
**Bahre Hailemariam**  
_Data Analyst & BI Developer_  
📧 [Email Adress](bahre.hail@gmail.com) | [Portfolio](https://bahre-hailemariam-data-analyst.crd.co/)| [LinkedIn]([https://www.linkedin.com/](https://www.linkedin.com/in/bahre-hailemariam/)) | [GitHub]([https://github.com/](https://github.com/BahreHailemariam))
