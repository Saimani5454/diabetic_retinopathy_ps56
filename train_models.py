# -*- coding: utf-8 -*-
"""train_models.ipynb

Updated for portability (works on Colab + Local PC)
"""

import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import lightgbm as lgb


# -----------------------------
# Handle dataset path
# -----------------------------
if os.path.exists("/content/pronostico_dataset (1).csv"):
    DATA_PATH = "/content/pronostico_dataset (1).csv"   # Colab path
else:
    DATA_PATH = r"C:\Users\saima\Downloads\diabetic retinopathy\pronostico_dataset (1).csv"  # Local path

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"❌ Dataset not found at: {DATA_PATH}\n"
        "Please place 'pronostico_dataset (1).csv' in the project folder."
    )

ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(DATA_PATH, sep=None, engine="python")
print("✅ Data Loaded Successfully")
print("Initial Data Shape:", df.shape)
print("Columns:", df.columns)

# Encode target column
df["prognosis"] = df["prognosis"].map({"no_retinopathy": 0, "retinopathy": 1})

X = df.drop(columns=["ID", "prognosis"], errors="ignore")
y = df["prognosis"]

# -----------------------------
# Train/Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler.pkl"))

# -----------------------------
# Models
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True),
    "XGBoost": xgb.XGBClassifier(eval_metric="logloss"),
    "LightGBM": lgb.LGBMClassifier()
}

metrics = {}

# -----------------------------
# Train + Evaluate
# -----------------------------
for name, model in models.items():
    print(f"🚀 Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    metrics[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }

    joblib.dump(model, os.path.join(ARTIFACTS_DIR, f"{name.replace(' ', '_').lower()}.pkl"))

# -----------------------------
# Save Metrics
# -----------------------------
metrics_df = pd.DataFrame(metrics).T
metrics_df.to_csv(os.path.join(ARTIFACTS_DIR, "metrics.csv"))
print("\n✅ Training completed. Metrics saved at:", os.path.join(ARTIFACTS_DIR, "metrics.csv"))
