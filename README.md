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
