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

df.totalcharges = pd.to_numeric(df.totalcharges, errors = 'coerce') #coerce is to ignore
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
categorical = ['gender', 'seniorcitizen', 'partner', 'dependents', 'phoneservice', 'multiplelines', 'internetservice',
       'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
       'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
       'paymentmethod']
print (df_full_train[categorical].nunique())

print ("\n\n\n")
print ("### 3.5 Feature importance: Churn rate and risk ratio")

# lets see if gender makes a difference
all_females = df_full_train[df_full_train.gender == 'female'].churn.mean()
all_males = df_full_train[df_full_train.gender == 'male'].churn.mean()
global_churn = df_full_train.churn.mean()
print (all_females, all_males, global_churn)

# now lets see partners
print(df_full_train.partner.value_counts())
w_partner = df_full_train[df_full_train.partner == 'yes'].churn.mean()
wo_partner = df_full_train[df_full_train.partner == 'no'].churn.mean()
print (w_partner, wo_partner, global_churn)
# looks like without partner churn more

print (global_churn - w_partner)
print (global_churn - wo_partner)

#this gives the idea that partner variable may be more important to predict churn than gender
#1) Difference: global-group, if >0 less likely to churn, but if <0 more likely to churn
#2) Risk ratio: group/gloval if >1 more likely to churn, if <1 les likely

rr_no_partner = wo_partner/global_churn #ratio is greather than 1
rr_partner = w_partner/global_churn 

# risk ratio gives information of how important a variable is 

# lets do this for all variable systematically
for c in categorical:
    df_group = df_full_train.groupby(c).churn.agg(['mean','count'])
    df_group['diff'] = df_group['mean'] - global_churn
    df_group['risk'] = df_group['mean'] / global_churn
    print (df_group)
    print ()

print ("\n\n\n")
print ("### 3.6 Feature importance: Mutual information")
#is a way to measure the importance of categorical values
# how much do we know about churn, when we check another variable

from sklearn.metrics import mutual_info_score
print(mutual_info_score(df_full_train.churn, df_full_train.contract)) #the closer to 0, the less we learn. So with contract we learn more than with gender
print(mutual_info_score(df_full_train.churn, df_full_train.gender)) 

def mutual_info_churn_score(series):
    return mutual_info_score(series, df_full_train.churn)

mi = df_full_train[categorical].apply(mutual_info_churn_score)
mi = mi.sort_values(ascending=False)
print (mi)
# we see that the ones in top are variables that give lots of info, so for machine learning, we should use these values to predict better


print ("\n\n\n")
print ("### 3.7 Feature importance: Correlation")
# here we will talk about numerical values
print(df_full_train[numerical])
print (df_full_train[numerical].corrwith(df_full_train.churn))

print (df_full_train[df_full_train.tenure <= 2].churn.mean())
print (df_full_train[(df_full_train.tenure > 2) & (df_full_train.tenure <= 12)].churn.mean())
print (df_full_train[df_full_train.tenure > 12].churn.mean())

print (df_full_train[df_full_train.monthlycharges <= 20].churn.mean())
print (df_full_train[(df_full_train.monthlycharges > 20) & (df_full_train.monthlycharges <= 50)].churn.mean())
print (df_full_train[df_full_train.monthlycharges > 50].churn.mean())

print (df_full_train[numerical].corrwith(df_full_train.churn).abs())
# we see that the highest correlation is in tenure, so it is the most important variable between the Numerical variables

print ("\n\n\n")
print ("### 3.8 One hot encoding")

# as example, we have gender and contract. Gender is F or M and contract is 2 years, 1 year or 2 months or less.
# this would become a matrix of 5 columns, where each row is a client and then we will have a 0 or 1 in female and male, and 0 or 1 in each of the 3 conracts

from sklearn.feature_extraction import DictVectorizer
#dicts = df_train[['gender', 'contract','tenure']].iloc[:100].to_dict(orient='records')
train_dicts = df_train[categorical+numerical].to_dict(orient='records')
#print (train_dicts)
dv = DictVectorizer(sparse=False) # this is a sparse matrix because it has many zeros, if we want to print sparse=False
dv.fit(train_dicts) 
print (dv.transform(train_dicts[:5])[0])
print (dv.get_feature_names_out())

X_train = dv.fit_transform(train_dicts) # we can do both steps above in 1

val_dicts = df_val[categorical+numerical].to_dict(orient='records')
X_val = dv.fit_transform(val_dicts) 

print ("\n\n\n")
print ("### 3.9 logistic regression")

def sigmoid (z):
    return 1/(1+np.exp(-z))
z = np.linspace(-5,5,51)
sigmoid(z)
#plt.plot(z,sigmoid(z))
#plt.show()

# we use it to get probability from a score
def linear_regression(xi):
    result = w0

    for j in range(len(w)):
        result = result + xi[j] * w[j]

    return result

def logistic_regression(xi):
    result = w0

    for j in range(len(w)):
        score = score + xi[j] * w[j]
    result = sigmoid(score)
    return result

print ("\n\n\n")
print ("### 3.10 train logistic regression with Scikit-Learn")

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
print (model.intercept_[0])
print (model.coef_[0].round(3))

# we can already predict
print (model.predict(X_train)) # these are hard prediction, 0 or 1
print (model.predict_proba(X_train)) #these are soft prediction, with probability. First column is pro of being 0 and second belonging to positive class. We are interested in the second, which is prob of churning.

y_pred = model.predict_proba(X_val)[:, 1]
print (y_pred)
print (y_pred >= 0.5)
churn_decision = (y_pred >= 0.5)
print (df_val[churn_decision].customerid) # these may churn so send them emails of discount :D

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

print (df_pred)
print (df_pred.correct.mean())


print ("\n\n\n")
print ("### 3.11 Model interpretation")

zip(dv.get_feature_names(), model.coef_[0].round(3)) # we will create tuples of coeficient and variable
small = ['contract','tenure','monthlycharges']
print(df_train[small].iloc[:10].to_dict(orient = "records"))
dicts_train_small = df_train[small].to_dict(orient = "records")
dicts_val_small = df_val[small].to_dict(orient = "records")
dv_small = DictVectorizer(sparse=False)
dv_small.fit(dicts_train_small)
print(dv_small.get_feature_names())
X_train_small = dv_small.transform(dicts_train_small)
model_small = LogisticRegression()
model_small.fit(X_train_small, y_train)
w0 = model_small.intercept_[0]
w = model_small.coef_[0]
print (w0,w)
print (dict(zip(dv_small.get_feature_names(),w.round(3))))


print ("\n\n\n")
print ("### 3.12 Using the Model")

dicts_full_train = df_full_train[categorical+numerical].to_dict(orient='records')
print (dicts_full_train[:3])

dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)
y_full_train = df_full_train.churn.values
model = LogisticRegression()
model.fit(X_full_train, y_full_train )
dicts_test = df_test[categorical+numerical].to_dict(orient='records')
X_test = dv.transform(dicts_test)
y_pred = model.predict_proba(X_test)[:,1]
churn_decision = (y_pred >= 0.5)
print ((churn_decision == y_test).mean())

customer = dicts_test[10]
X_small = dv.transform([customer])
print (model.predict_proba(X_small)) # this person has small probability to churn
print (y_test) # and here we get 0, which he was not gonna churn so it is accurate


