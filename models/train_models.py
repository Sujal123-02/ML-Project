"""
Student Placement Prediction — Model Training & Saving Script
=============================================================
Trains all 4 classification models (KNN, Decision Tree, Random Forest, XGBoost)
on the placement dataset, evaluates them, and persists the fitted objects for
the Flask backend to load at runtime.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "placementdata.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load & Pre-process
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

# Encode categorical columns the same way for ALL models
categorical_cols = ["ExtracurricularActivities", "PlacementTraining", "PlacementStatus"]
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # save for inverse-transform later

# Features / target
feature_cols = [
    "CGPA",
    "Internships",
    "Projects",
    "Workshops/Certifications",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "ExtracurricularActivities",
    "PlacementTraining",
    "SSC_Marks",
    "HSC_Marks",
]
X = df[feature_cols]
y = df["PlacementStatus"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# StandardScaler (needed for KNN; we apply it to all base-models for fairness)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 2. Define models
# ---------------------------------------------------------------------------
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False,
    ),
}

# Which models need scaled features?
scaled_models = {"KNN", "Decision Tree"}

# ---------------------------------------------------------------------------
# 3. Train, evaluate, persist
# ---------------------------------------------------------------------------
results = {}

for name, model in models.items():
    X_tr = X_train_scaled if name in scaled_models else X_train
    X_te = X_test_scaled if name in scaled_models else X_test

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    results[name] = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,
    }

    print(f"\n{'='*40}")
    print(f"Model: {name}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"Confusion Matrix:\n{np.array(cm)}")
    print(classification_report(y_test, y_pred))

# ---------------------------------------------------------------------------
# 4. Save everything
# ---------------------------------------------------------------------------
bundle = {
    "models": models,
    "scaler": scaler,
    "feature_cols": feature_cols,
    "label_encoders": label_encoders,
    "results": results,
    "scaled_models": scaled_models,
}
with open(os.path.join(ARTIFACTS_DIR, "models_bundle.pkl"), "wb") as f:
    pickle.dump(bundle, f)

print("\n[OK] All models trained and saved to", ARTIFACTS_DIR)
