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




