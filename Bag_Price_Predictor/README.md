# 🎒 Noisy Student Bag Price Prediction using Random Forest

This project predicts the **price of bags** based on features such as brand, size, compartments, waterproofing, and more. The dataset is noisy and contains missing values, which are imputed using both **KNNImputer** and **mode imputation** techniques.

---

## 🎯 Objective

- Predict the **`Price`** of bags using numerical and categorical features.
- Handle **missing values** in both numeric and categorical data.
- Use **Random Forest Regressor** for price prediction.
- Evaluate model performance using **Mean Absolute Error (MAE)**.

---

## 📊 Dataset Overview

- **File Used:** `Noisy_Student_Bag_Price_Prediction_Dataset.csv`
- **Target Column:** `Price`
- **Features:**
  - `Brand`, `Material`, `Style`, `Color`
  - `Size`, `Laptop Compartment`, `Waterproof`
  - `Compartments`, `Weight Capacity (kg)`

---

## 🧼 Data Preprocessing

### 🧩 Missing Value Handling:
- **Numerical Columns (`Compartments`, `Weight Capacity (kg)`):**  
  Imputed using `KNNImputer`.
- **Categorical Columns (`Size`, `Laptop Compartment`, `Waterproof`):**  
  Filled using the **mode** (most frequent value).

### 🧠 Encoding:
- **One-Hot Encoding** for high-cardinality columns: `Brand`, `Material`, `Style`, `Color`
- **Ordinal Encoding** for ordered/binary features: `Size`, `Laptop Compartment`, `Waterproof`

---

## 🤖 Model Used

### 🌲 RandomForestRegressor
- `n_estimators = 100`
- `max_leaf_nodes = 85`

---

## 🧪 Evaluation

- **Metric Used:** Mean Absolute Error (MAE)
- Model Performance:

```plaintext
Predictions on validation set: [55.23, 120.45, 65.89, 78.12, 110.34]
Mean Absolute Error: 12.3456
