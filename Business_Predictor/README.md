# 📊 Coffee Shop Latte Price Prediction (Random Forest & Gradient Boosting)

This project predicts **latte prices** based on business, demographic, and competitive factors. The goal is to identify the most profitable zip codes for opening new coffee shops by analyzing factors such as population, existing shop count, customer ratings, and income levels.

---

## 🎯 Objective

- Predict **latte prices** using regression models
- Identify **top zip codes** to target for new shop expansion using:
  - High population
  - Low coffee shop competition
  - Low average ratings
  - High median salary

---

## 🧾 Datasets

### 1. `CH_2_Coffee Shop data.xlsx`
- Coffee shop details like:
  - `Business Name`, `City`, `Zip Code`
  - `Rating`, `Gender Majority`, `Median Salary`, `Latte Price`

### 2. `population.csv`
- Zip code level demographic data including:
  - `Total Population`
  - `Household Distribution`

---

## ⚙️ Preprocessing

1. **Join Datasets**
   - Merge coffee shop and population data on 5-digit zip codes
2. **Feature Engineering**
   - Calculate `CoffeeShopCount` per zip code
   - Select features: `Rating`, `Median Salary`, `Population`, `CoffeeShopCount`
3. **Scaling**
   - Apply `StandardScaler` to input features
4. **Train-Test Split**
   - 80% training / 20% testing (`random_state=42`)

---

## 🧠 Models Used

### 1. 🌲 Random Forest Regressor
- `n_estimators`: Tuned via GridSearchCV (50, 100, 200)
- `max_depth`: [None, 10, 20]

### 2. 🚀 Gradient Boosting Regressor
- `n_estimators`: [50, 100, 200]
- `learning_rate`: [0.01, 0.1, 0.2]
- `max_depth`: [3, 5, 10]

### 3. 📈 Linear Regression
- Used for baseline comparison

---

## 📐 Evaluation Metrics

- **Mean Absolute Error (MAE)**  
- **Mean Squared Error (MSE)**  
- **R² Score**  

---

## 🧪 Results

