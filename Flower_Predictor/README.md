# 🌸 Iris Species Classification (Random Forest & XGBoost)

This project uses the classic **Iris dataset** to classify flower species based on their sepal and petal measurements. The dataset is modeled using two popular ensemble learning algorithms: **Random Forest Classifier** and **XGBoost Classifier**.

---

## 🎯 Objective

- Classify iris flowers into three species: `setosa`, `versicolor`, and `virginica`
- Compare model performance between **RandomForestClassifier** and **XGBClassifier**
- Evaluate the models using **accuracy**, **MAE**, and **cross-validation**

---

## 📊 Dataset Overview

- **Source:** `seaborn.load_dataset('iris')`
- **Target Variable:** `species`
- **Features:**
  - `sepal_length`
  - `sepal_width`
  - `petal_length`
  - `petal_width`

---

## ⚙️ Preprocessing

1. **Target Encoding:**
   - `species` column is one-hot encoded using `OneHotEncoder`.

2. **Train-Test Split:**
   - 80% training, 20% validation

---

## 🧠 Models Used

### 1. 🎲 RandomForestClassifier
- `n_estimators = 100`
- `random_state = 0`

### 2. ⚡ XGBClassifier
- `n_estimators = 500`
- `learning_rate = 0.05`

---

## 📈 Evaluation Metrics

- **Accuracy Score:** Measures the percentage of correct predictions
- **Mean Absolute Error (MAE):** Measures the average absolute error
- **Cross-Validation (5-Fold Accuracy):** Ensures model generalization

---

## 🧪 Results

```text
Random Forest Accuracy      : 0.9667
XGBoost Accuracy            : 0.9667

Random Forest MAE           : 0.0333
XGBoost MAE                 : 0.0333

Random Forest CV Accuracy   : 0.9600
XGBoost CV Accuracy         : 0.9667
