#Sep 24

import numpy as np
import pandas as pd

# download data from: https://www.kaggle.com/CooperUnion/cardataset

### 2.2 data preparation 

df = pd.read_csv("data.csv")
df.head()

# we first remove spaces and make all lowercase in columns
df.columns = df.columns.str.lower().str.replace(" ","_")

# the same with values
# first check the type of columns, only change names to strings (object)
strings = list(df.dtypes[df.dtypes == 'object'].index)
#strings a python list containing the name of the columns with strings
for col in strings:
    df[col] = df[col].str.lower().str.replace(" ","_")

### 2.3 exploratory data analysis
for col in df.columns:
    print (col)
    print (df[col].unique()[:5])
    print (df[col].nunique())
    print ()

import matplotlib.pyplot as plt
import seaborn as sns

#sns.histplot(df.msrp, bins=50)
#plt.show()

# lets zoom out to see the long tail of distribution

#sns.histplot(df.msrp[df.msrp < 100000], bins=50)
#plt.show()

# this kind of long tail withh confuse our model. It is not good for ML. So we want to get rid of the tail.
# we can log the data

#log1p adds 1 to all values so we dont do log of 0
price_logs = np.log1p(df.msrp)
#sns.histplot(price_logs, bins=50)
#plt.show()
#this shape is more normal, which is better for ML

# we also have nans in the data so we will remove them.
print (df.isnull().sum()) #shows how many missing values

### 2.4  setting up the validation framework.

# we want, 60% for training, 20% for validation and 20% for testing
n = len(df)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test # we do this so we use all the rest for training

# to make sure it is reproducible, we set the seed.
np.random.seed(2)

# we want to shuffle the data and not draw it in order
idx = np.arange(n)
np.random.shuffle(idx)

df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train+n_val]]
df_test = df.iloc[idx[n_train+n_val:]]

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

print (df_train)
print (len(df_train),len(df_val),len(df_test))

y_train = np.log1p(df_train.msrp.values)
y_val = np.log1p(df_val.msrp.values)
y_test = np.log1p(df_test.msrp.values)

# we want to remove this variable so we dont do the mistake of not doing it and get a perfect score
del df_train['msrp']
del df_val['msrp']
del df_test['msrp']

print (len(y_train))

### 2.5 Linear repgression
print (2.5)

xi = [453,11,86]
w0 = 7.17
w = [0.01,0.04,0.002]

def linear_regression(xi):
    n = len(xi) 

    pred = w0

    for j in range(n):
        pred = pred + w[j] * xi[j]
    return pred

print (linear_regression(xi))
print (np.expm1(12.312)) # the price would be this

### 2.6 Linear regression vector form
print ("\n\n\n")
print ("### 2.6")

def dot(xi,w):
    n = len(xi)
    res = 0.0
    for j in range(n):
        res = res + xi[j] * w[j]
    return res

#def linear_regression(xi):
#    return w0 + dot(xi,w)

w_new = [w0] + w # we simplify the function, because the first feature if it is 1, w0 becomes 0

def linear_regression(xi):
    xi = [1] + xi
    return dot(xi,w_new)

print (linear_regression(xi))


xi = [453,11,86]
w0 = 7.17
w = [0.01,0.04,0.002]

x1 = [1, 148, 24, 1385]
x2 = [1, 132, 25, 2031]
x10 = [1, 453, 11, 86]

X = [x1, x2, x10]
X = np.array(X) # makes list of list intro matrix

print (X.dot(w_new))

print ("\n\n\n")
print ("### 2.7 Training a linear regression model")

X = [
    [148, 24, 1385],
    [132, 25, 2031],
    [453, 11, 86],
    [158, 24, 185],
    [172, 25, 201],
    [413, 11, 86],
    [38, 54, 185],
    [142, 25, 431],
    [453, 31, 86],
]
y = [10000, 20000, 15000, 20050, 10000, 20000, 15000, 25000, 12000]

def train_linear_regression(X, y):
    X = np.array(X) # makes list of list intro matrix
    ones = np.ones(X.shape[0])
#to add the bias term
    X =  np.column_stack([ones, X])
    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    return w_full[0], w_full[1:]

print (train_linear_regression(X,y))

print ("\n\n\n")
print ("### 2.8 Car price baseline model")

base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']
X_train = df_train[base].fillna(0).values
w0, w = train_linear_regression (X_train,y_train)
print (w0, w)
y_pred = w0 + X_train.dot(w)

#sns.histplot(y_pred, color="red", alpha=0.5, bins=50)
#sns.histplot(y_train,color="blue", alpha=0.5, bins=50)
#plt.show()

#shape of prediction is not so good, it predicts smaller value than the target. But we need to have a objective way of evaluating


print ("\n\n\n")
print ("### 2.9 RMSE")

def rmse (y, y_pred):
    error = y - y_pred
    se = error ** 2
    mse = se.mean()
    return np.sqrt(mse)

print (rmse(y_train, y_pred))


print ("\n\n\n")
print ("### 2.10 Validating the model")

# function that prepares the data regardless if it is validation, train or test
def prepare_X (df):
    df_num = df[base]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X

X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)
print (rmse(y_val, y_pred))



print ("\n\n\n")
print ("### 2.11 Simple feature engineering")

# add more features to the training
# like Year, which is very important for the price
# but we dont want prepare_X to change the original data.

def prepare_X (df):
    df = df.copy()
    df['age'] = 2017 - df.year # 2017 is newest, so 0 would be new and big value old
    features = base + ['age']
    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X

X_train = prepare_X(df_train)
X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)
print (rmse(y_val, y_pred))

#sns.histplot(y_pred, color="red", alpha=0.5, bins=50)
#sns.histplot(y_val,color="blue", alpha=0.5, bins=50)
#plt.show()
#not the distribution is better, but we can improve it.


print ("\n\n\n")
print ("### 2.12 Categorical Variables")
