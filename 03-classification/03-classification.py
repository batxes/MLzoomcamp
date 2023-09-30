# Sept 26

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print ("\n\n\n")
print ("### 3.1 Churn prediction project")

# https://www.kaggle.com/datasets/blastchar/telco-customer-churn
data = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(data)

print ("\n\n\n")
print ("### 3.2 Data preparation")

df.columns = df.columns.str.lower().str.replace(' ', '_')
categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ','_')

print (df.T)
print (df.dtypes)
print (df.totalcharges) #most are numbers but some are not because the data is object
#pd.to_numeric(df.totalcharges) will fail

tc = pd.to_numeric(df.totalcharges, errors = 'coerce') #coerce is to ignore
print (df[tc.isnull()][['customerid','totalcharges']]) #value is missing at totalcharges for this customers

df.totalcharges = df.totalcharges.fillna(0)


df.churn = (df.churn == "yes").astype(int) #converts yes to 1 and no to 0

print (df.churn)


print ("\n\n\n")
print ("### 3.3 Setting up the validation framework")

from sklearn.model_selection import train_test_split

# train_test_split? # this shows docs in notebooks

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1) # this splits into test and train, but we also want to do validation, so we split again, and 20% of 80% is actually 25%
print (len(df_full_train),len(df_test))
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
print (len(df_train),len(df_test),len(df_val))
df_train = df_train.reset_index(drop=True)
df_full_train = df_full_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
y_train = df_train.churn.values
y_test = df_test.churn.values
y_val = df_val.churn.values
del df_train['churn']
del df_test['churn']
del df_val['churn']

print ("\n\n\n")
print ("### 3.4 EDA, exploratory data analysis")
print (df_full_train.isnull().sum())
print (df_full_train.churn.value_counts(normalize=True))
print ("Global churn rate is how many people churn, which is 0.26. We can also get to that with the mean")
print (df_full_train.churn.mean())
global_churn_rate = df_full_train.churn.mean()

print (df_full_train.dtypes)
print ("There are 3 numerical values: tenure, monthly charges and total charges.")
numerical = ["tenure","monthlycharges","totalcharges"]
categorical = ['customerid', 'gender', 'seniorcitizen', 'partner', 'dependents', 'phoneservice', 'multiplelines', 'internetservice',
       'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
       'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
       'paymentmethod']
print (df_full_train[categorical].nunique())

print ("\n\n\n")
print ("### 3.5 Feature importance: Churn rate and risk ratio")
