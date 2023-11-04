import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score

df = pd.read_csv("weatherAUS.csv",sep=",")
df.RainToday = (df.RainToday == "Yes").astype(int) #converts yes to 1 and no to 0
df.RainTomorrow = (df.RainTomorrow == "Yes").astype(int) #converts yes to 1 and no to 0

# Date could be changed into day, month and year, which would boost the prediction

months_dic = {
    1:'january',2:'feabruary',3:'march',4:'april',5:'may',6:'june',
    7:'july',8:'august',9:'september',10:'october',11:'november',12:'december'
}

def extract_date(x):
    years, months, days = [],[],[]
    for d in x.Date.values:
        year,month,day = d.split("-")
        years.append(int(year))
        months.append(int(month))
        days.append(int(day))
    return years, months, days

years, months, days = extract_date(df)
df["year"] = years
df["month"] = months
df["day"] = days
df.month = df.month.map(months_dic)
del df["Date"]

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

df_full_train = df_full_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_full_train = df_full_train.RainTomorrow.values
y_test = df_test.RainTomorrow.values

del df_full_train['RainTomorrow']
del df_test['RainTomorrow']

def get_type(df):
    categorical = [col for col in df.columns if df[col].dtype == 'O'] #0 = Object
    numerical = [col for col in df.columns if df[col].dtype != 'O']
    
    return categorical, numerical

categorical, numerical = get_type(df_full_train)

print('Categorical features:', categorical)
print('Numerical features:', numerical)

for df1 in [df_full_train, df_test]:
    df1['WindGustDir'].fillna(df1['WindDir9am'], inplace=True)
    df1['WindDir3pm'].fillna(df1['WindDir9am'], inplace=True)
    
    df1['WindGustDir'].fillna(df1['WindDir3pm'], inplace=True)
    df1['WindDir9am'].fillna(df1['WindDir3pm'], inplace=True)
    
    df1['WindDir3pm'].fillna(df1['WindGustDir'], inplace=True)
    df1['WindDir9am'].fillna(df1['WindGustDir'], inplace=True)
    
for df1 in [df_full_train, df_test]:
    df1["WindGustDir"].fillna(df_full_train["WindGustDir"].mode()[0], inplace=True)
    df1["WindDir9am"].fillna(df_full_train["WindGustDir"].mode()[0], inplace=True)
    df1["WindDir3pm"].fillna(df_full_train["WindGustDir"].mode()[0], inplace=True)

# for the rest of the values, fill nans with median values since we have some outliers.
for df1 in [df_full_train, df_test]:
    for col in numerical:
        median_value = df_full_train[col].median() #use median of the full train
        df1[col].fillna(median_value, inplace=True) 
    

dicts_full_train = df_full_train.to_dict(orient="records")
dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)

dicts_test = df_test.to_dict(orient="records")
X_test = dv.transform(dicts_test)

#for XGboost
dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=dv.get_feature_names())
dtest = xgb.DMatrix(X_test, feature_names=dv.get_feature_names())


xgb_params = {
        'eta': 0.1, #how fast it trains
        'max_depth': 20,
        'min_child_weight': 1,  #te same as min_samples_leaf
        'objective': 'binary:logistic', #becasue we have a binary class type
        'eval_metric':'auc',
        'nthread': 8,
        'seed': 1,
        'verbosity':1,
        }
model = xgb.train(xgb_params, dfulltrain, num_boost_round=500)
y_pred = model.predict(dtest)
auc = roc_auc_score(y_test, y_pred)
print (" XGBoost AUC: {}".format(auc))

import pickle

output_file = 'model_xgb_rain_prediction.bin'

f_out = open(output_file,'wb')
pickle.dump((dv,model),f_out) # we also need the dv to understand later the prediction, not only the model. SO we put both in a tuple
f_out.close()

print ("model saved")



