# Student Placement Prediction — ML Project

A full-stack Machine Learning project that predicts whether a student will be **Placed** or **Not Placed** using four classification models. The project includes a Flask backend, a professional single-page dashboard, and a beautiful claymorphism-styled prediction interface.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Models Used](#models-used)
3. [Dataset Description](#dataset-description)
4. [Project Structure](#project-structure)
5. [Tech Stack](#tech-stack)
6. [Setup & Installation](#setup--installation)
7. [How to Run](#how-to-run)
8. [API Endpoints](#api-endpoints)
9. [Frontend Pages](#frontend-pages)
10. [Model Performance Summary](#model-performance-summary)
11. [Screenshots](#screenshots)
12. [Questions & Notes](#questions--notes)

---

## Project Overview

This project demonstrates end-to-end machine learning pipeline:

- **Data preprocessing** — label encoding of categorical features, standard scaling for distance-based models.
- **Model training** — 4 classification models are trained on the same dataset with the same train/test split (`80/20`, `random_state=42`).
- **Model evaluation** — Accuracy, Precision, Recall, F1-Score, and Confusion Matrix computed for every model.
- **Prediction API** — A Flask REST API accepts student data via JSON and returns predictions from all 4 models.
- **Visual Dashboard** — A single HTML page with Chart.js visualizations for model comparison (bar charts, radar chart, confusion matrices).

---

## Models Used

| # | Model | Type | Description |
|---|-------|------|-------------|
| 1 | **K-Nearest Neighbors (KNN)** | Base Model | Distance-based classifier with `n_neighbors=5`, uses `StandardScaler` for feature scaling. |
| 2 | **Decision Tree** | Base Model | Tree-based classifier with `random_state=42`, uses `StandardScaler` for consistency. |
| 3 | **Random Forest** | Ensemble Model | Bagging-based ensemble with `n_estimators=100`, `random_state=42`. |
| 4 | **XGBoost** | Ensemble Model | Gradient boosting framework, `eval_metric='logloss'`, `random_state=42`. |

---

## Dataset Description

**File:** `models/placementdata.csv`  
**Size:** 10,000 records × 12 columns  
**Source:** Student placement dataset

| Column | Type | Description |
|--------|------|-------------|
| `StudentID` | int | Unique student identifier (dropped during training) |
| `CGPA` | float | Cumulative Grade Point Average (0–10) |
| `Internships` | int | Number of internships completed |
| `Projects` | int | Number of projects completed |
| `Workshops/Certifications` | int | Number of workshops or certifications |
| `AptitudeTestScore` | int | Score in aptitude test (0–100) |
| `SoftSkillsRating` | float | Soft skills rating (0–5) |
| `ExtracurricularActivities` | str | Yes / No |
| `PlacementTraining` | str | Yes / No |
| `SSC_Marks` | int | SSC (10th) marks |
| `HSC_Marks` | int | HSC (12th) marks |
| `PlacementStatus` | str | **Target** — `Placed` / `NotPlaced` |

---

## Project Structure

```
Project/
├── backend/
│   └── app.py                  # Flask backend (serves API + frontend)
├── frontend/
│   └── index.html              # Single-page UI (prediction + dashboard)
├── models/
│   ├── train_models.py         # Training script for all 4 models
│   ├── placementdata.csv       # Dataset
│   └── artifacts/
│       └── models_bundle.pkl   # Saved trained models + scaler + metrics
├── Base_Models.ipynb           # Original notebook: KNN + Decision Tree
├── Ensemble_Learning.ipynb     # Original notebook: Random Forest + XGBoost
├── placementdata.csv           # Original dataset (copy in root)
└── README.md                   # This file
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| **ML Models** | scikit-learn, XGBoost |
| **Backend** | Flask (Python) |
| **Frontend** | HTML, CSS (Claymorphism), JavaScript |
| **Charts** | Chart.js |
| **Font** | Google Inter |
| **Data** | pandas, numpy |
| **Model Persistence** | pickle |

---

## Setup & Installation

### Prerequisites

- **Python 3.8+** installed
- **pip** package manager

### Install Dependencies

```bash
pip install flask scikit-learn xgboost pandas numpy
```

---

## How to Run

### Step 1: Train the Models

> Only needed once (or whenever you want to retrain).

```bash
cd models
python train_models.py
```

This will:
- Load & preprocess `placementdata.csv`
- Train all 4 models
- Evaluate each model
- Save everything to `models/artifacts/models_bundle.pkl`

### Step 2: Start the Flask Server

```bash
cd backend
python app.py
```

### Step 3: Open the App

Open your browser and go to: **http://127.0.0.1:5000**

You will see two pages:
1. **Prediction** — Enter student details and see predictions from all 4 models.
2. **Dashboard** — Compare model accuracy, precision, recall, F1-score with interactive charts and confusion matrices.

---

## API Endpoints

### `POST /api/predict`

Accepts student features, returns predictions from all 4 models.

**Request Body (JSON):**
```json
{
  "CGPA": 8.5,
  "Internships": 2,
  "Projects": 3,
  "Workshops_Certifications": 2,
  "AptitudeTestScore": 85,
  "SoftSkillsRating": 4.5,
  "ExtracurricularActivities": "Yes",
  "PlacementTraining": "Yes",
  "SSC_Marks": 80,
  "HSC_Marks": 85
}
```

**Response:**
```json
{
  "predictions": {
    "KNN": { "prediction": "Placed", "confidence": 80.0 },
    "Decision Tree": { "prediction": "Placed", "confidence": 100.0 },
    "Random Forest": { "prediction": "Placed", "confidence": 87.0 },
    "XGBoost": { "prediction": "Placed", "confidence": 92.34 }
  }
}
```

### `GET /api/dashboard`

Returns evaluation metrics for all models.

**Response:**
```json
{
  "metrics": {
    "KNN": {
      "accuracy": 0.772,
      "precision": 0.7263,
      "recall": 0.721,
      "f1_score": 0.7236,
      "confusion_matrix": [[947, 225], [231, 597]]
    },
    ...
  }
}
```

---

## Frontend Pages

### 1. Prediction Page
- Clean light-blue + white claymorphism-styled form
- Enter all 10 student features
- Click "Predict Placement" to get results from all 4 models
- Each model card shows prediction (Placed / Not Placed), confidence percentage, and a visual progress bar
- Liquid blob background decorations for a modern feel

### 2. Dashboard Page
- **Stat Cards** — Quick view of each model's accuracy (best model highlighted)
- **Accuracy Bar Chart** — Vertical bar comparison of all models
- **Precision / Recall / F1 Grouped Bar Chart** — Side-by-side comparison
- **Radar Chart** — Overall performance profile overlay
- **F1-Score Ranking** — Horizontal bar chart sorted by F1-score
- **Confusion Matrices** — Color-coded tables (green = correct, red = incorrect)

---

## Model Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| KNN | 77.20% | 72.63% | 72.10% | 72.36% |
| Decision Tree | 72.65% | 66.36% | 68.84% | 67.58% |
| Random Forest | **78.15%** | **74.65%** | 71.50% | **73.04%** |
| XGBoost | 77.85% | 74.03% | **71.62%** | 72.81% |

> **Best Overall Model: Random Forest** (highest accuracy and F1-score)

---

## Questions & Notes

1. **Why do KNN and Decision Tree use StandardScaler but Random Forest and XGBoost don't?**
   - KNN relies on distance metrics, so feature scaling is essential. Decision Tree doesn't technically need it, but we apply it for consistency with the original notebook's approach. Random Forest and XGBoost are tree-based ensemble methods that are scale-invariant, so we follow the original notebook's approach of not scaling for them.

2. **Can I add more models?**
   - Yes! Simply add the model to the `models` dictionary in `train_models.py`, retrain, and the frontend/backend will automatically pick it up.

3. **How is the confidence percentage calculated?**
   - It uses `predict_proba()` from scikit-learn/XGBoost, which returns the probability of each class. The confidence shown is the probability of the predicted class.

4. **Can the dataset be changed?**
   - Yes, as long as the new CSV has the same column names and data types. Place it in `models/` and retrain.

5. **What Python version is required?**
   - Python 3.8 or newer is recommended.

---

**Made with ML and Flask**
