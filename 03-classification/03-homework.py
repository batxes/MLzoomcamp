import pandas as pd

df = pd.read_csv('data.csv',sep=",")
print (df)
print (df.columns)


data = df[['Make', 'Model', 'Year', 'Engine HP',
       'Engine Cylinders', 'Transmission Type', 'Vehicle Style',
       'highway MPG', 'city mpg', 'MSRP']]
data = data.rename(columns={'MSRP':'price'})

data.columns = data.columns.str.replace(' ', '_').str.lower()
data = data.fillna(0)
print (data.columns)

print ("### question 01")
print ()
print ()
print (data.transmission_type.value_counts())


print ("### question 02")
print ()
print ()
print (data.dtypes)
numerical = [ 'year', 'engine_hp',
       'engine_cylinders', 
       'highway_mpg', 'city_mpg']
categorical = ['make', 'model','vehicle_style','transmission_type']

import matplotlib.pyplot as plt

print (data.corr())
#plt.matshow(data.corr())
#plt.show()

mean_price = data.price.mean()

def price_binary(row):
    if row.price > mean_price:
        return 1
    else:
        return 0

data["above_average"] = data.apply(price_binary,axis=1)
print (data)

from sklearn.model_selection import train_test_split
df_full_train, df_test = train_test_split(data, test_size=0.2, random_state=42) # this splits into test and train, but we also want to do validation, so we split again, and 20% of 80% is actually 25%
print (len(df_full_train),len(df_test))
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)
print (len(df_train),len(df_test),len(df_val))
df_train = df_train.reset_index(drop=True)
df_full_train = df_full_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
y_train = df_train.above_average.values
y_test = df_test.above_average.values
y_val = df_val.above_average.values
del df_train['above_average']
del df_test['above_average']
del df_val['above_average']

print ("### question 03")
print ()
print ()
from sklearn.metrics import mutual_info_score

def mutual_info_churn_score(series):
    return mutual_info_score(series, df_full_train.above_average)

mi = df_full_train[categorical].apply(mutual_info_churn_score)
mi = mi.sort_values(ascending=False).round(2)
print (mi)

print ("### question 04")
print ()
print ()
from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer(sparse=False) 
train_dicts = df_train[categorical+numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dicts) # we can do both steps above in 1
dv = DictVectorizer(sparse=False) 
val_dicts = df_val[categorical+numerical].to_dict(orient='records')
X_val = dv.fit_transform(val_dicts) 

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='liblinear', C=10, max_iter=1000, random_state=42)
model.fit(X_train, y_train)

print ("asdadasda")

y_pred = model.predict(X_val)
print (y_pred)

# to see how accurate our predicitons are:
# how much of these are the same?
print (y_val)
print (churn_decision.astype(int))
print ((y_val == churn_decision).mean()) #80% pretty good

df_pred = pd.DataFrame()
df_pred['probability'] = y_pred
df_pred['prediction'] = churn_decision.astype(int)
df_pred['actual'] = y_val
df_pred['correct'] = df_pred.prediction == df_pred.actual

