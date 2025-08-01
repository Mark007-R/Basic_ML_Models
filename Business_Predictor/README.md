# ☕ Latte Price Prediction in Top 5 Zip Codes

This project predicts latte prices in the top 5 zip codes using various machine learning regression models. The prediction is based on features like average income, population, and gender demographics per zip code.

---

## 🎯 Objective

- Predict the price of a latte in different zip codes.
- Use regression models such as Linear Regression, Random Forest Regressor, and XGBoost Regressor.
- Apply predictions to the top 5 zip codes based on the dataset.

---

## 🛠️ Tech Stack

- Python 3.12
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib / Seaborn (optional for visualization)

---

## 📁 Dataset Features

| Column Name         | Description                                |
|---------------------|--------------------------------------------|
| `Latte Price`       | Target variable (float)                    |
| `Zip Code`          | Numeric zip code of the location           |
| `Average Income`    | Avg. income in the zip code area           |
| `Population`        | Total population of the area               |
| `Male Population`   | Male population                            |
| `Female Population` | Female population                          |
| `Business Name`     | Name of the coffee business (ignored)      |
| `Street address`    | Address (ignored)                          |
| `City`, `State`     | City and State (ignored)                   |
| `Phone`, `ID no.`   | Metadata (ignored)                         |
| `Gender majority`   | Gender dominance in area (ignored)         |

Only the **numeric and demographic features** were used for prediction.

---

## 🧠 Models Used

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

Models were trained and tuned using **GridSearchCV** with `neg_mean_squared_error` as the scoring metric.

---