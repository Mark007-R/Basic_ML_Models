# 🔭 Gamma vs. Hadron Particle Classification

This project focuses on classifying high-energy particles detected by the **MAGIC gamma-ray telescope**. Using a dataset of particle events, we apply several machine learning models to predict whether each event was caused by a **gamma ray** or a **hadron**.

---

## 🧠 Project Goal

To build a machine learning classifier that can distinguish between **gamma ray** and **hadron** events based on telescope data.

---

## 🗃️ Dataset Overview

- **Source:** telescope_data.csv
- **Rows:** ~19,000 particle events
- **Features (10):**
  - `fLength`, `fWidth`, `fSize`, `fConc`, `fConc1`
  - `fAsym`, `fM3Long`, `fM3Trans`, `fAlpha`, `fDist`
- **Target:**
  - `class` → 0 = Gamma, 1 = Hadron

---

## 📦 Libraries Used

- `numpy`, `pandas` – data manipulation
- `matplotlib`, `seaborn` – data visualization
- `scikit-learn` – model training and evaluation
- `imblearn` – for handling class imbalance (oversampling)

---

## 📊 Exploratory Data Analysis

- Correlation heatmap to understand relationships between features.
- Histograms to visualize how features differ by class (Gamma vs. Hadron).
- Class imbalance observed → addressed with **RandomOverSampler**.

---

## 🧪 Preprocessing

1. Encode the `class` label using `OrdinalEncoder`.
2. Split data into Train (60%), Validation (20%), Test (20%).
3. Normalize feature values using `StandardScaler`.
4. Apply **oversampling** on the training set only.

---

## 🏋️ Model Training & Evaluation

Trained and evaluated the following models:

| Model               | Accuracy | F1 Score (Hadron) | Notes                            |
|---------------------|----------|-------------------|----------------------------------|
| **K-Nearest Neighbors (K=3)** | 81% | 0.73 | Simple, performs decently |
| **Naive Bayes**      | 74% | 0.52 | Struggles with Hadron class |
| **Logistic Regression** | 80% | 0.72 | Balanced and interpretable |
| **Support Vector Machine (SVM)** | **87%** | **0.81** | ⭐ Best performer |

---

## 🧠 Key Insights

- **SVM outperformed** other models with an 87% accuracy and balanced classification of both classes.
- **Class imbalance** significantly impacted performance until oversampling was applied.
- Simple preprocessing steps (scaling, encoding) had a big impact on model performance.

---

## 🛠️ How to Run

1. Clone the repo or download the code.
2. Place `telescope_data.csv` in the same directory.
3. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn
