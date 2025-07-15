# 🚢 Titanic Survival Prediction (XGBoost Model)

This project uses the Titanic dataset to predict passenger survival based on various demographic and travel-related features. It leverages preprocessing pipelines and the powerful XGBoost classification algorithm.

---

## 🎯 Objective

To build a machine learning pipeline that:
- Handles missing values and categorical features
- Trains a predictive model using `XGBoostClassifier`
- Evaluates model performance using **accuracy**, **mean absolute error**, and **cross-validation**

---

## 🗂️ Dataset Overview

- **File:** `titanic.csv`
- **Target Variable:** `survived`
- **Features Used:**
  - Numerical: `age`, `fare`, `sibsp`, `parch`, etc.
  - Categorical: `sex`, `embarked`, `class`, `who`, `embark_town`

---

## ⚙️ Data Preprocessing

1. **Numerical Features:**
   - Split into:
     - With missing values → imputed with **constant value (0)**
     - Without missing values → passed through unchanged

2. **Categorical Features:**
   - Imputed with most frequent value
   - Encoded using **One-Hot Encoding** (with unknown handling)

3. **Dropped Columns:**
   - Uninformative or redundant: `Unnamed: 0`, `deck`, `alive`

---

## 🤖 Model Training

- **Algorithm:** `XGBoostClassifier`
- **Parameters:**
  - `n_estimators = 500`
  - `learning_rate = 0.05`
  - `eval_metric = logloss`
- **Pipeline Components:**
  - `ColumnTransformer` for preprocessing
  - `Pipeline` for full integration with the model

---

## 🧪 Model Evaluation

- **Train-Test Split:** 80% train, 20% test
- **Metrics:**
  - ✅ **Accuracy Score:** Measures overall correctness
  - 📉 **Mean Absolute Error (MAE):** Measures average error
  - 🔁 **Cross-Validation Score (CV-MAE):** 5-fold CV

### 📈 Results

```text
Mean Absolute Error: 0.1100
Validation Accuracy: 0.8652
Cross Validation: 0.1401
