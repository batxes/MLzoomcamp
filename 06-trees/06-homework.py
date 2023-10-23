import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("housing.csv",sep=",")
df = df.query('ocean_proximity == "<1H OCEAN" or ocean_proximity == "INLAND"')
df.columns = df.columns.str.lower()
df = df.fillna(0)
df["median_house_value"] = np.log10(df.median_house_value)
print (df)

from sklearn.model_selection  import train_test_split

df_full_train, df_test = train_test_split (df, test_size = 0.2, random_state= 1)
df_train, df_val = train_test_split (df_full_train, test_size = 0.25, random_state= 1)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
y_train = df_train.median_house_value.values
y_val = df_val.median_house_value.values
y_test = df_test.median_house_value.values

del df_train['median_house_value']
del df_val['median_house_value']
del df_test['median_house_value']

from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score

train_dicts = df_train.to_dict(orient='records')
dv = DictVectorizer(sparse=True)
X_train = dv.fit_transform(train_dicts)
dt = DecisionTreeRegressor(max_depth=1)
dt.fit(X_train, y_train)
from sklearn.tree import export_text

print ("QUESTION 1:")
print (export_text(dt,feature_names = dv.get_feature_names()))

dt = DecisionTreeRegressor(n_estimators=10,random_state = 1,n_jobs=-1)
dt.fit(X_train, y_train)

def rmse (y, y_pred):
    error = y - y_pred
    se = error ** 2
    mse = se.mean()
    return np.sqrt(mse)

print (rmse(y_val, y_pred))




