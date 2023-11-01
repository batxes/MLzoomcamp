# October 9, 2023

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("data.csv",sep=",")
df = df[['Make', 'Model', 'Year', 'Engine HP','Engine Cylinders', 'Transmission Type','Vehicle Style', 'highway MPG', 'city mpg', 'MSRP']].copy()
df.columns = df.columns.str.lower().str.replace(' ', '_')
df = df.fillna(0)
numerical = ['year', 'engine_hp', 'engine_cylinders','highway_mpg', 'city_mpg']
categorical = ['make','model','transmission_type','vehicle_style']

mean_price = df.msrp.mean()

def binary_price(row):
    if row.msrp >= mean_price:
        return 1
    else:
        return 0

df["above_average"] = df.apply(binary_price, axis=1)
del  df["msrp"]

# split the data

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1) 
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.above_average.values
y_test = df_test.above_average.values
y_val = df_val.above_average.values

del df_train['above_average']
del df_test['above_average']
del df_val['above_average']

print ("\n\n")
print ("Question 1: ROC AUC Feature importance \n\n")

for n in numerical:
    auc = roc_auc_score(y_train, df_train[n].values)
    if auc < 0.5:
        auc = roc_auc_score(y_train, -df_train[n].values)
    print ("{} auc score: {}".format(n,auc))

print ("Question number 1 answer: engine_hp")


print ("\n\n")
print ("Question 2: Training the model\n\n")


train_dicts = df_train.to_dict(orient='records')
dv = DictVectorizer()
X_train = dv.fit_transform(train_dicts)

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
model.fit(X_train, y_train)

val_dicts = df_val.to_dict(orient='records')
X_val = dv.transform(val_dicts)

y_pred = model.predict_proba(X_val)[:, 1]
     
print (roc_auc_score(y_val, y_pred).round(3))

print ("Question number 2 answer: 1.0")


print ("\n\n")
print ("Question 3: Precision and Recall\n\n")

scores = []

#thresholds = np.arange(0, 1, 0.01)
thresholds = np.linspace(0, 1, 101)


for t in thresholds:
    actual_positive = (y_val == 1)
    actual_negative = (y_val == 0)

    predict_positive = (y_pred >= t)
    predict_negative = (y_pred < t)

    tp = (predict_positive & actual_positive).sum()
    tn = (predict_negative & actual_negative).sum()

    fp = (predict_positive & actual_negative).sum()
    fn = (predict_negative & actual_positive).sum()

    scores.append((t, tp, fp, fn, tn))

columns2 = ['threshold', 'tp', 'fp', 'fn', 'tn']
df_scores = pd.DataFrame(scores, columns=columns2)

df_scores['p'] = df_scores.tp / (df_scores.tp + df_scores.fp)
df_scores['r'] = df_scores.tp / (df_scores.tp + df_scores.fn)
     
plt.plot(df_scores.threshold, df_scores.p, label='precision')
plt.plot(df_scores.threshold, df_scores.r, label='recall')

plt.legend()
plt.show()

print ("Answer to question 3: between 0.48")


print ("\n\n")
print ("Question 4: F1 Score\n\n")



df_scores['f1'] = 2 * ((df_scores.p*df_scores.r)/ (df_scores.p + df_scores.r))
print (df_scores.head(10))
print (df_scores.loc[df_scores.f1.argmax()])

plt.figure(figsize=(10, 5))

plt.plot(df_scores.threshold, df_scores.f1)
plt.vlines(0.52, 0.45, 0.9, color='grey', linestyle='--', alpha=0.5)

plt.xticks(np.linspace(0, 1, 11))
plt.show()

print ("Answer to Q4: 0.52")


print ("\n\n")
print ("Question 5: 5-Fold CV\n\n")

from sklearn.model_selection import KFold
columns = list(df.columns)
columns.remove('above_average')

def train(df_train, y_train, C=1.0):
    dicts = df_train[columns].to_dict(orient='records')

    dv = DictVectorizer()
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
    model.fit(X_train, y_train)

    return dv, model

def predict(df, dv, model):
    dicts = df[columns].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

kfold = KFold(n_splits=5, shuffle=True, random_state=1)

scores=[]

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.above_average.values
    y_val = df_val.above_average.values

    dv, model = train(df_train, y_train, C=1.0)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

print('%.3f +- %.3f' % (np.mean(scores), np.std(scores)))

#print (np.mean(scores),np.std(scores))
print('%.3f +- %.3f' % (np.mean(scores), np.std(scores)))


print ("Answer to Q5: less than 0.00001")


print ("\n\n")
print ("Question 6: Hyperparameter Tunning \n\n")


from tqdm.auto import tqdm

kfold = KFold(n_splits=5, shuffle=True, random_state=1)

for C in [0.01, 0.1, 0.5, 10]:
    scores = []

    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_train = df_train.above_average.values
        y_val = df_val.above_average.values

        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)

    print('C=%4s, %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))

print ("Answer to Q6: 10.")

