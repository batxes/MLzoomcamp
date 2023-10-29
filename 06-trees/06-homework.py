import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("housing.csv",sep=",")
df = df.query('ocean_proximity == "<1H OCEAN" or ocean_proximity == "INLAND"')
df.columns = df.columns.str.lower()
df = df.replace("<","-",regex=True)
df = df.fillna(0)
df["median_house_value"] = np.log(df.median_house_value)
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
print (export_text(dt,feature_names = dv.get_feature_names_out()))

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=10, random_state = 1)
rf.fit(X_train, y_train)
val_dicts = df_val.to_dict(orient='records')
X_val = dv.transform(val_dicts)
y_pred = rf.predict(X_val)

def rmse (y, y_pred):
    error = y - y_pred
    se = error ** 2
    mse = se.mean()
    return np.sqrt(mse)

print ("QUESTION 2:")
print (rmse(y_val, y_pred))

scores = []

for n in range(10, 201, 10):
    print (n)
    rf = RandomForestRegressor(n_estimators=n, random_state = 1, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    error = rmse(y_val, y_pred)
    scores.append((n,error))
df_scores = pd.DataFrame(scores, columns=['n_estimators','rmse'])
plt.plot(df_scores.n_estimators, df_scores.rmse)
plt.show()

print ("QUESTION 3: 160")

scores = []

for d in [10, 15, 20, 25]:
    for n in range(10, 201, 10):
        print (n, d)
        rf = RandomForestRegressor(n_estimators=n, random_state = 1, n_jobs=-1,max_depth=d)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        error = rmse(y_val, y_pred)
        scores.append((n,d,error))
df_scores = pd.DataFrame(scores, columns=['n_estimators','max_depth','rmse'])

for d in [10, 15, 20, 25]:
    df_subset = df_scores[df_scores.max_depth == d]
    plt.plot(df_subset.n_estimators, df_subset.rmse,label='max_depth = %s' % d)
plt.legend()
plt.show()
print ("QUESTION 4: max_depth 20")

rf = RandomForestRegressor(n_estimators=10, random_state = 1, n_jobs=-1,max_depth=20)
rf.fit(X_train, y_train)
for feat, importance in zip(df_train.columns, rf.feature_importances_):
    print ('feature: {f}, importance: {i}'.format(f=feat, i=importance))

print ("QUESTION 5: total_bedrooms")


import xgboost as xgb

features = dv.get_feature_names_out()
print (features)
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(features))
dval = xgb.DMatrix(X_val, label=y_val, feature_names=list(features))
watchlist = [(dtrain, 'train'),(dval,'dval')]
scores = {}
for etas in [0.3, 0.1]:
    print ("ETA: {}".format(etas))
    xgb_params = {
            'eta': etas, #how fast it trains
            'max_depth': 6,
            'min_child_weight': 1,  #te same as min_samples_leaf
            'objective': 'reg:squarederror', 
            'nthread': 8,
            'seed': 1,
            'verbosity':1,
            }
    model = xgb.train(xgb_params, dtrain, num_boost_round=100,evals=watchlist,verbose_eval=5)
    y_pred = model.predict(dval)


print ("QUESTION 6: 0.1")
