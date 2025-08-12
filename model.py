import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 📁 Define dataset path using os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
DATA_PATH = os.path.join(DATASET_DIR, "heart_disease_data.csv")  # ✅ Correct filename here

# 🗂️ Load and clean the dataset
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please check the path.")

disease_data = pd.read_csv(DATA_PATH, encoding='latin1')
disease_data.columns = disease_data.columns.str.replace('ï»¿', '', regex=False)

# ✅ Validate required column
if 'target' not in disease_data.columns:
    raise ValueError("Expected column 'target' not found in dataset.")

# 🔍 Data preprocessing
x = disease_data.drop(['target'], axis=1)
y = disease_data['target']

# ✅ Check for missing values
if x.isnull().any().any() or y.isnull().any():
    raise ValueError("Dataset contains missing values. Please clean the data before training.")

# ✅ Ensure numeric types
if not np.issubdtype(y.dtype, np.number):
    raise TypeError("Target column must be numeric.")

# 🔀 Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=2)

# 🔧 Feature scaling
scaler = StandardScaler()
x_train_scaled_df = pd.DataFrame(scaler.fit_transform(x_train), columns=x.columns)
x_test_scaled_df = pd.DataFrame(scaler.transform(x_test), columns=x.columns)

# 🧠 Model training
model_scaled = LogisticRegression(max_iter=1000)
model_scaled.fit(x_train_scaled_df, y_train)

# 📈 Model evaluation
x_train_acc = accuracy_score(model_scaled.predict(x_train_scaled_df), y_train)
x_test_acc = accuracy_score(model_scaled.predict(x_test_scaled_df), y_test)

# ✅ Export for app.py
__all__ = [
    "x", "scaler", "model_scaled", "x_train_acc", "x_test_acc", "disease_data"
]