# 🩺 Breast Cancer Diagnosis using Logistic Regression

This project uses the Breast Cancer Wisconsin Diagnostic dataset to classify tumors as **malignant** or **benign** using a **Logistic Regression** model. It includes preprocessing, correlation analysis, hyperparameter tuning, and performance evaluation.

---

## 📊 Dataset

- **Source**: `sklearn.datasets.load_breast_cancer()`
- **Samples**: 569
- **Features**: 30 numerical features extracted from breast mass images
- **Target Classes**:
  - `0`: Malignant
  - `1`: Benign

---

## 🔧 Libraries Used

```bash
pandas
numpy
scikit-learn
matplotlib
seaborn

## 📈 Exploratory Data Analysis

- Checked for missing values and data types using `.info()`
- Explored summary statistics using `.describe()`
- Visualized the correlation matrix using a Seaborn heatmap to understand feature relationships

---

## 🔄 Preprocessing

- All features were standardized to have a mean of 0 and standard deviation of 1
- The dataset was split into training and testing sets (80% training, 20% testing)

---

## 🤖 Model: Logistic Regression

- Implemented a Logistic Regression model to classify tumors as malignant or benign

---

## ✅ Cross-Validation

- Performed 5-fold cross-validation using negative mean squared error as the scoring metric
- Cross-validation showed consistent and low error across all folds, indicating good generalization

---

## 🔍 GridSearchCV for Hyperparameter Tuning

- Grid search was used to find the best hyperparameters based on F1-score
- The best parameters found were:
  - `C`: 0.1
  - `penalty`: l2
  - `solver`: liblinear
  - `max_iter`: 100
- Best cross-validation F1-score: **0.9827**

---

## ✅ Accuracy

- The model achieved an accuracy of **0.9912** on the test set, indicating high performance on unseen data
