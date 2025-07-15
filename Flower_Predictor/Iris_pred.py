import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

data = sns.load_dataset('iris')
og_data = pd.DataFrame(data=data)

X = pd.DataFrame(og_data.copy())
y = pd.DataFrame(og_data.species)
X.drop(['species'], inplace=True, axis=1)

one_hot_e = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
new_y = pd.DataFrame(one_hot_e.fit_transform(y))

train_X, valid_X, train_y, valid_y = train_test_split(X, new_y, train_size=0.8, test_size=0.2, random_state=0)

model1 = RandomForestClassifier(n_estimators=100, random_state=0)
model2 = XGBClassifier(n_estimators=500, learning_rate=0.05)

model1.fit(train_X, train_y)
model2.fit(train_X, train_y)

pred1 = model1.predict(valid_X)
pred2 = model1.predict(valid_X)

acs1 = accuracy_score(pred1, valid_y)
acs2 = accuracy_score(pred2, valid_y)
print(acs1)
print(acs2)

mae1 = mean_absolute_error(pred1, valid_y)
mae2 = mean_absolute_error(pred2, valid_y)
print(mae1)
print(mae2)

cvs1 = cross_val_score(model1, X, new_y, cv=5, scoring='accuracy')
cvs2 = cross_val_score(model2, X, new_y, cv=5, scoring='accuracy')
print(cvs1.mean())
print(cvs2.mean())