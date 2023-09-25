import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('housing.csv',sep=",")

#sns.histplot(df.median_house_value, bins=50)
#plt.show()
df = df.query('ocean_proximity == "<1H OCEAN" or ocean_proximity == "INLAND"')
print (df.ocean_proximity.nunique())
df = df[["latitude","longitude","housing_median_age","total_rooms","total_bedrooms","population","households","median_income","median_house_value"]]

print ("\n\n\n")
print ("### Q1")
print (df.isnull().sum())

print ("\n\n\n")
print ("### Q2")
print (np.median(df.population))

np.random.seed(42)

n = len(df)                 
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test

idx = np.arange(n)
np.random.shuffle(idx)                     

df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train+n_val]]
df_test = df.iloc[idx[n_train+n_val:]]

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = np.log1p(df_train.median_house_value.values)
y_val = np.log1p(df_val.median_house_value.values)
y_test = np.log1p(df_test.median_house_value.values)

print ("\n\n\n")
print ("### Q3")

def train_linear_regression(X, y):
    X = np.array(X) 
    ones = np.ones(X.shape[0])
    X =  np.column_stack([ones, X])
    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    return w_full[0], w_full[1:]

base = df.columns.values.tolist()
X_train = df_train[base].fillna(0).values
w0, w = train_linear_regression (X_train,y_train)
y_pred = w0 + df_val.dot(w)

def rmse (y, y_pred):
    error = y - y_pred
    se = error ** 2
    mse = se.mean()
    return np.sqrt(mse)

print (rmse(y_val, y_pred))

X_train = df_train[base].fillna(df_train.median_house_value.mean()).values
w0, w = train_linear_regression (X_train,y_train)
y_pred = w0 + df_val.dot(w)
print (rmse(y_val,y_pred))
print ("\n\n\n")
print ("### Q4")
X_train = df_train[base].fillna(0).values
X_val = df_val[base].fillna(0).values
X_test = df_test[base].fillna(0).values

def train_linear_regression_reg(X, y, r=0.001):
    X = np.array(X) # makes list of list intro matrix
    ones = np.ones(X.shape[0])
    X =  np.column_stack([ones, X])
    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])  #<--------
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    return w_full[0], w_full[1:]


for r in [0.0, 0.000001, 0.0001, 0.001, 1000]:
    w0, w = train_linear_regression_reg(X_train, y_train, r)
    y_pred = w0 + X_val.dot(w)
    score = rmse(y_val, y_pred)
    print (r, w0, round(score,2))

print ("\n\n\n")
print ("### Q5")

scores = []
for seed in np.arange(10):
    idx = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(idx)                     
    df_train = df.iloc[idx[:n_train]]
    df_val = df.iloc[idx[n_train:n_train+n_val]]
    df_test = df.iloc[idx[n_train+n_val:]]

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    X_train = df_train[base].fillna(0).values
    X_val = df_val[base].fillna(0).values
    X_test = df_test[base].fillna(0).values

    y_train = np.log1p(df_train.median_house_value.values)
    y_val = np.log1p(df_val.median_house_value.values)
    y_test = np.log1p(df_test.median_house_value.values)

    base = df.columns.values.tolist()
    w0, w = train_linear_regression (X_train,y_train)
    y_pred = w0 + X_val.dot(w)
    score = rmse(y_val, y_pred)
    scores.append(score)
print (scores)
print (round(np.std(scores),3))

print ("\n\n\n")
print ("### Q6")

idx = np.arange(n)
np.random.seed(9)
np.random.shuffle(idx)                     
df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train+n_val]]
df_test = df.iloc[idx[n_train+n_val:]]
df_full_train = pd.concat([df_train, df_val])                         

df_train = df_train.reset_index(drop = True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_full_train = df_full_train.reset_index(drop = True)

X_val = df_val[base].fillna(0).values
X_test = df_test[base].fillna(0).values
X_full_train = df_full_train[base].fillna(0).values

y_train = np.log1p(df_train.median_house_value.values)
y_val = np.log1p(df_val.median_house_value.values)
y_test = np.log1p(df_test.median_house_value.values)
y_full_train = np.concatenate([y_train, y_val])

w0, w = train_linear_regression_reg(X_full_train, y_full_train, r=0.001)
y_pred = w0 + X_test.dot(w)
score = rmse(y_test,y_pred)
print (score)




