import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# first day: started 20:45- ended 22:00 = 1.45 minutes
print( "6. Decision Trees and Ensemble Learning") 
print ()
print( "6.1 Cedit Risk scoring Project")
print ()

# From https://github.com/gastonstat/CreditScoring/blob/master/CreditScoring.csv
dataset = "CreditScoring.csv"

print( "6.2 Data Cleaning and preparation")
print ()

df = pd.read_csv(dataset,sep=",")

# We have many categorical values encoded as numbers, so we want them to contain words to udnerstad the data
# we can see where the categories are if we go to the website where the data is.
# missing values are cnoded as 999999 

df.columns = df.columns.str.lower()
status_values = {
        1:'ok',
        2:'default',
        0:'unk'
        }
home_values = {
        1: 'rent',
        2: 'owner',
        3: 'private',
        4: 'ignore',
        5: 'parents',
        6: 'other',
        7: 'unk'
        }
marital_values = {
        1: 'single',
        2: 'married',
        3: 'widow',
        4: 'separated',
        5: 'divorced',
        6: 'unk'
        }
records_values = {
        1:'no',
        2:'yes',
        0:'unk'
        }
job_values = {
        1:'fixed',
        2:'partime',
        3:'freelance',
        4:'others',
        0:'unk'
        }
df.status = df.status.map(status_values)
df.home = df.home.map(home_values)
df.marital = df.marital.map(marital_values)
df.records = df.records.map(records_values)
df.job = df.job.map(job_values)
print (df.describe().round()) # we see income, assests and debt with lots ofg 999999 which are missing values
print (df.status.value_counts())
#df.status also has one unknown value, so we remove it.

df.income = df.income.replace(to_replace=99999999,value=np.nan)
df.assets = df.assets.replace(to_replace=99999999,value=np.nan)
df.debt = df.debt.replace(to_replace=99999999,value=np.nan)

df = df[df.status != 'unk'].reset_index(drop=True)

#@ do the train test and val split

from sklearn.model_selection  import train_test_split

df_full_train, df_test = train_test_split (df, test_size = 0.2, random_state= 11)
df_train, df_val = train_test_split (df_full_train, test_size = 0.25, random_state= 11)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
# remove the default variable from the data, and we want to convert it into a "1", because it is the TRUE people that are not paying and we want to know who not to give the credit
y_train = (df_train.status == 'default').astype('int').values
y_val = (df_val.status == 'default').astype('int').values
y_test = (df_test.status == 'default').astype('int').values
del df_train['status']
del df_val['status']
del df_test['status']

print (df_train)


print( "6.3 Decision trees")
print ()

# decision tree as a stracture that has tree like network with different conditions and True or False decisions

def assess_risk(client):
    if client['records']  == 'yes':
        if client['job'] == 'yes':
            return "ok"
        else:
            return "no"
    else:
        return "no"

# this would be a representation, but we want the model to learn it automatically so we can use it afterwards.

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score

train_dicts = df_train.fillna(0).to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

val_dicts = df_val.fillna(0).to_dict(orient='records')
X_val = dv.transform(val_dicts)
y_pred = dt.predict_proba(X_val)[:, 1]
print ("validation: " ,roc_auc_score(y_val, y_pred)) #this is 0.66
y_pred = dt.predict_proba(X_train)[:, 1]
print ("train: ",roc_auc_score(y_train, y_pred)) # this is 1, which is overfitting. It memorizes the data but can not predict well new data.
# if the tree is very long and train a lot, in the end it creates many trees that are specific to each client and then it memorizes all.
# But if we have trees that are only 3 or 4 of depth, it should create good rules and not memorize

dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X_train, y_train)
val_dicts = df_val.fillna(0).to_dict(orient='records')
X_val = dv.transform(val_dicts)
y_pred = dt.predict_proba(X_val)[:, 1]
print ("validation: " ,roc_auc_score(y_val, y_pred))
y_pred = dt.predict_proba(X_train)[:, 1]
print ("train: ",roc_auc_score(y_train, y_pred)) 

#now with a lower depth, the performance gets much better. 

#dt = DecisionTreeClassifier(max_depth=1)
#dt.fit(X_train, y_train)
#val_dicts = df_val.fillna(0).to_dict(orient='records')
#X_val = dv.transform(val_dicts)
#y_pred = dt.predict_proba(X_val)[:, 1]
#print ("validation: " ,roc_auc_score(y_val, y_pred))
#y_pred = dt.predict_proba(X_train)[:, 1]
#print ("train: ",roc_auc_score(y_train, y_pred)) 
##in fact, with a depth = 1, called decision stump, it is very similar to the whole dataset.

#to visualize trees we can use:
from sklearn.tree import export_text
print (export_text(dt,feature_names = dv.get_feature_names()))

print( "6.4 Decision tree Learning Algorithm")
print ()
# Here we will learn how the decision tree come up with rules

data = [
    [8000,'default'],
    [2000,'default'],
    [0,'default'],
    [5000,'ok'],
    [5000,'ok'],
    [4000,'ok'],
    [9000,'ok'],
    [3000,'default'],
]

df_example = pd.DataFrame(data,columns=['assets','status'])
print (df_example) # we want to use assets to train our decision tree
# we will split our data into default and ok depending on a cutoff of the assets column
print (df_example.sort_values('assets')) # we could put the cutoff in 2000, 3000, 4000, 5000 and 8000
Ts = [2000, 3000, 4000, 5000, 8000]
#for each T, we cut our dataset and see which is the best cutoff
for T in Ts:
    df_left = df_example[df_example.assets <= T]
    df_right = df_example[df_example.assets > T]
    print (T)
    #print (df_left)
    print (df_left.status.value_counts(normalize=True))
    #print (df_right)
    print (df_right.status.value_counts(normalize=True))
    print ()
    print ("---------")

# now we need to know which is the best split.
# we can evaluate with missclassification of each predicttion
# This miss classification is called impurity. So we can create a table of impurity for each df.
# best T is 3000 ebcause it has the lowest impurity: lowest number of oks in the let and lowest number of defaultss in the right. Oks in the left 0%, defautls in the right 20%. Then we calculate the average and we get a 10%. This number is the loweest when we hae T of 3000.

# now imaging we add another feature.


data = [
    [8000,3000,'default'],
    [2000,1000,'default'],
    [0,1000,'default'],
    [5000,1000,'ok'],
    [5000,1000,'ok'],
    [4000,1000,'ok'],
    [9000,500,'ok'],
    [3000,2000,'default'],
]

df_example = pd.DataFrame(data,columns=['assets','debt','status'])
print (df_example.sort_values('debt')) 
thresholds = {
        'assets': [2000, 3000, 4000, 5000, 8000],
        'debt': [500, 1000, 2000],
        }
for feature, Ts in thresholds.items():
    print (feature)
    for T in Ts:
        df_left = df_example[df_example[feature] <= T]
        df_right = df_example[df_example[feature] > T]
        print (T)
        #print (df_left)
        print (df_left.status.value_counts(normalize=True))
        #print (df_right)
        print (df_right.status.value_counts(normalize=True))
        print ()
        print ("---------")

# Here, even if we use debt, the impurity is the lowest with assets = 3000
# This is how the algorithm lears which are the best cutoffs for each variable, but we need to now how to stop because it can get huge. When we get the max or min and still split then it does not make sense to keep going and we need to stop. If the group is already pure, stop splitting.
# another stopping criteria is the max depth
# and another one is when the group os too small
# using these criteria, we can prevent on overfitting
"""
Decision tree learning algorithm:

    - Find the best split
    - Stop if max_depth is reached
    - if left is sufficiently large
        and not pure:
        -> repeat for left
    - if right is sufficiently large
        and not pure:
        -> repeat for right
"""


print( "6.5 Decision trees parameter tuning")
print ()

