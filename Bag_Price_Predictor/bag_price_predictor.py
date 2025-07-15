from sklearn.impute import KNNImputer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

# Load the dataset
data = pd.read_csv('Noisy_Student_Bag_Price_Prediction_Dataset.csv')

# Drop duplicate rows and reset the index
data = data.drop_duplicates().reset_index(drop=True)

# Impute missing values in numeric columns using KNN
numeric_cols = ['Compartments', 'Weight Capacity (kg)']
imputer = KNNImputer()
for col in numeric_cols:
    data[col] = imputer.fit_transform(data[[col]])

# Impute missing values in categorical columns with mode
cat_mode_cols = ['Size', 'Laptop Compartment', 'Waterproof']
for col in data.select_dtypes(include='object').columns:
    data[col] = data[col].fillna(data[col].mode()[0])

# One-hot encode selected categorical columns
one_hot_cols = ['Brand', 'Material', 'Style', 'Color']
encoder_oh = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_oh = encoder_oh.fit_transform(data[one_hot_cols])
df_one_hot = pd.DataFrame(encoded_oh)

# Ordinal encode other categorical columns
ordinal_cols = ['Size', 'Laptop Compartment', 'Waterproof']
encoder_ord = OrdinalEncoder()
encoded_ord = pd.DataFrame(encoder_ord.fit_transform(data[ordinal_cols]),
                           columns=ordinal_cols,
                           index=data.index)

# Combine all features: numeric, one-hot encoded, and ordinal encoded
df_features = pd.concat([data[numeric_cols], encoded_ord, df_one_hot], axis=1)

# Add target column to the features DataFrame temporarily for NA filtering
df_features['Price'] = data['Price']

# Drop rows with any remaining missing values
df_features.dropna(inplace=True)

# Separate target and features
y = df_features['Price'].reset_index(drop=True)
X = df_features.drop('Price', axis=1).reset_index(drop=True)
X.columns = X.columns.astype(str)

# Train-test split
train_X, valid_X, label_y, valid_y = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0
)

# Train and evaluate a Random Forest model using MAE
def get_score(train_X, valid_X, label_y, valid_y):
    rf_model = RandomForestRegressor(n_estimators=100, max_leaf_nodes=85)
    rf_model.fit(train_X, label_y)
    pred_val = rf_model.predict(valid_X)
    # Show first 5 predictions
    print("Predictions on validation set:", pred_val[:5])
    return mean_absolute_error(valid_y, pred_val)


# Get Mean Absolute Error
mae = get_score(train_X, valid_X, label_y, valid_y)
print(f"Mean Absolute Error: {mae:.4f}")