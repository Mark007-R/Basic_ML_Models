import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

original_data = pd.read_csv('titanic.csv')

val_d = [col for col in original_data.columns if original_data[col].dtype in [
    'float64', 'int64']]

val_d1 = [col for col in original_data.columns if original_data[col].dtype in [
    'float64', 'int64'] and original_data[col].isnull().any()]

cat_d = [col for col in original_data.columns if original_data[col].dtype ==
         'object' and original_data[col].nunique() <= 10]

val_data = original_data[val_d].copy()
val_data.drop(['survived', 'Unnamed: 0'], axis=1, inplace=True)
val_d = val_data.columns.tolist()

cat_data = original_data[cat_d].copy()
cat_data.drop(['deck', 'embark_town', 'alive'], axis=1, inplace=True)
cat_d = cat_data.columns.tolist()

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
cat_data_1 = pd.DataFrame(OH_encoder.fit_transform(cat_data))
cat_data_1.index = cat_data.index

my_impute = SimpleImputer(strategy='median')
val_data[val_d1] = pd.DataFrame(my_impute.fit_transform(original_data[val_d1]))
val_data[val_d1].columns = original_data[val_d1].columns

y = original_data['survived']

X = pd.concat([val_data, cat_data_1], axis=1)
X.columns = X.columns.astype(str)

# X.to_csv('titanic_new.csv')

train_X, valid_X, label_y, valid_y = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0)


def get_score(train_X, valid_X, label_y, valid_y):
    rf_model = RandomForestClassifier(
        n_estimators=95, max_leaf_nodes=100, random_state=0)
    rf_model.fit(train_X, label_y)
    pred_val = rf_model.predict(valid_X)
    return mean_absolute_error(pred_val, valid_y), accuracy_score(pred_val, valid_y)


mae, acs = get_score(train_X, valid_X, label_y, valid_y)
print(f"Mean Absolute Error: {mae:.4f}")
print(f"Validation Accuracy: {acs:.4f}")
