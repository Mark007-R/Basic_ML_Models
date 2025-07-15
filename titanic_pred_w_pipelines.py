import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
# from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
# imported the titanic csv file
original_data = pd.read_csv('titanic.csv')
# seperated the int and float valued columns name
val_d = [col for col in original_data.columns if original_data[col].dtype in [
    'float64', 'int64']]
# seperated the int and float valued columns name which contains null values
val_d1 = [col for col in original_data.columns if original_data[col].dtype in [
    'float64', 'int64'] and original_data[col].isnull().any()]
# seperated the object valued columns name
cat_d = [col for col in original_data.columns if original_data[col].dtype ==
         'object' and original_data[col].nunique() <= 10]
# copied the int n float valued data in another df
val_data = original_data[val_d].copy()
# droping unecessary columns
val_data.drop(['Unnamed: 0', 'survived'], axis=1, inplace=True)
# copied the object valued data in another df
cat_data = original_data[cat_d].copy()
# removed the unnecessary object data from category data
cat_data.drop(['deck', 'alive'], axis=1, inplace=True)
# again put all the int and float columns name in variable(val_d) which are necessary
val_d = val_data.columns.tolist()
# seperated the int and float column name which are not having null values
val_d_no_missing = [col for col in val_d if col not in val_d1]
# again put all the object columns name in variable(cat_d) which are necessary
cat_d = cat_data.columns.tolist()
# now the preprocessing(imputation,one-hot-encoding)
n_data = SimpleImputer(strategy='constant')
c_data = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant')),
    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(transformers=[
    ('num_impute', n_data, val_d1),
    ('num_pass', n_data, val_d_no_missing),
    ('cat', c_data, cat_d)
])
# now the model(algorithm-RFC or XGBC) building
# model = RandomForestClassifier(n_estimators=100, random_state=0)
model = XGBClassifier(n_estimators=500, learning_rate=0.05, eval_metric='logloss')
pred = Pipeline(steps=[
    ('pp', preprocessor),
    ('md', model)
])
# giving the data in X
X = original_data[val_d+cat_d].copy()
# giving label in y
y = original_data['survived']
# spliting the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, train_size=0.8, random_state=0)
# fitting the data
pred.fit(X_train, y_train)
# predicting the data
p_val = pred.predict(X_test)
# calculating the mean absolute error
mae = mean_absolute_error(p_val, y_test)
# calculating the accuracy score
acs = accuracy_score(y_test, p_val)
# calculating the cross validation score
css = -1 * cross_val_score(pred, X, y, cv=5, scoring='neg_mean_absolute_error')
cssv = css.mean()
print(f"Mean Absolute Error: {mae:.4f}")
print(f"Validation Accuracy: {acs:.4f}")
print(f"Cross Validation: {cssv:.4f}")
