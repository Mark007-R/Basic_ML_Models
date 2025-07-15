# 🚖 Taxi Trip Price Prediction

A regression project using machine learning to predict taxi trip prices based on various ride features such as distance, time, traffic, weather, and fare structure.

---

## 🎯 Objective

To predict the final **Trip Price** for a taxi ride using given features like trip distance, fare rates, duration, traffic, and weather conditions.

---

## 📊 Dataset Overview

- **File:** `taxi_trip_pricing.csv`
- **Records:** 1,000 taxi trip samples
- **Features (before processing):**
  - `Trip_Distance_km`, `Trip_Duration_Minutes`, `Passenger_Count`
  - `Time_of_Day`, `Day_of_Week`, `Traffic_Conditions`, `Weather`
  - `Base_Fare`, `Per_Km_Rate`, `Per_Minute_Rate`
  - `Trip_Price` (target)

---

## 🛠️ Preprocessing Steps

1. **Missing Values:**
   - Numerical columns: Handled using **KNN Imputer**.
   - Categorical columns: Filled with the **mode**.

2. **Feature Engineering:**
   - Created:
     - `total_trip_distance_rate = Trip_Distance_km × Per_Km_Rate`
     - `total_duration_rate = Trip_Duration_Minutes × Per_Minute_Rate`
   - Dropped:
     - Raw rate and distance/duration columns to reduce multicollinearity.

3. **Encoding:**
   - Used **Ordinal Encoding** for categorical features: `Time_of_Day`, `Day_of_Week`, `Traffic_Conditions`, `Weather`.

---

## 🔍 Exploratory Data Analysis

- **Distributions:** Used histograms to explore feature spread and detect skewness.
- **Correlation:** Strong positive correlation between:
  - `Trip_Distance_km` and `Trip_Price` → **0.85**
  - `total_trip_distance_rate` and `Trip_Price`
- **Visuals:**
  - Histograms for numeric columns.
  - Heatmaps for feature correlations.

---

## 🤖 Model Training

- **Model Used:** `RandomForestRegressor`
- **Hyperparameters:**
  - `n_estimators=85`
  - `max_leaf_nodes=100`
  - `random_state=0`
- **Train-Test Split:** 80/20

---

## 📈 Results

- **Model Score (R²):** `0.91`
- **Mean Absolute Error (MAE):** `6.04`
- **Root Mean Squared Error (RMSE):** `12.54`

These results indicate a **highly accurate model** with low average error per prediction.

---

## 🌿 Feature Importance

Top contributing features:-
total_trip_distance_rate
total_duration_rate
Passenger_Count
Traffic_Conditions
Base_Fare


Plotted using a horizontal bar chart.

---

## 🧪 How to Run

1. Clone the repo or copy the code.
2. Ensure `taxi_trip_pricing.csv` is in your working directory.
3. Install dependencies:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
