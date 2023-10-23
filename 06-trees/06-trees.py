import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


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

# we can select max_depth and min_samples_leaf parameters tuning them
# min_samples_leaf is the minimum number of entries in each of the nodes, in our case default and ok
for d in [1,2,3,4,5,6,10, 15, 20, None]:
    dt = DecisionTreeClassifier(max_depth=d)
    dt.fit(X_train, y_train)
    y_pred = dt.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    print ('%4s -> %.3f' % (d, auc))

# looks like the best deppth is for 4, 5 and 6. # if we have to choose we can go for the lowest because it is easier to undertand

scores = []

for d in [4,5,6]:
    for s in [1, 2, 5, 10, 15, 20, 100, 200, 500]:
        dt = DecisionTreeClassifier(max_depth=d, min_samples_leaf = s)
        dt.fit(X_train, y_train)
        y_pred = dt.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        scores.append((d,s,auc))
        print ('%4s %s -> %.3f' % (d,s, auc))

# we have 6, 15 as highest score: 0.785
columns = ["max_depth", "min_samples_leaf", "auc"]
df_scores = pd.DataFrame(scores, columns=columns)
df_scores.sort_values(by="auc",ascending=True)
df_scores_pivot= df_scores.pivot(index='min_samples_leaf',columns=["max_depth"],values="auc")
print (df_scores_pivot)
#sns.heatmap(df_scores_pivot,annot=True, fmt='.3f')
#plt.show()

# here we started with depth and then filtered by a few values, but it could be that max depth of 8 works better with other min sample leafs. In this dataset we can try but for bigger datasets it is not so nice to tune all of them, so it is nice to tune big datasets by a value and the try the rest


dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf = 15)
dt.fit(X_train, y_train)
print( "6.6 Ensembles and random forest")
print ()
# random forest is basically putting together many of these decision trees
# imaging that we have a board of experts, like 5 people, and they all decide if we give credit or not.
# so random forest is the same but with models, each expert being a model
# then from each model we have a score and we can aggregate the score of each model(decision tree) creating a random forest
# we dont want to train the same model X times, but rather, we get a different set of features from teh client , instead of all of them which could be 10 features, we get 7 for example, and those 7 are different for each mode that we train

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=10, random_state = 1)
rf.fit(X_train, y_train)
y_pred = rf.predict_proba(X_val)[:, 1]
print(roc_auc_score(y_val, y_pred))
# we get a very good score without doing any tunning
# actually, there is a randomization in the training, so every time we run it again, we will get a slighlty different result. SO it is good for reprocudicbility to fix the seed

# what happens if we modify the number of models (n_ewstimators)?

#scores = []
#
#for n in range(10, 201, 10):
#    rf = RandomForestClassifier(n_estimators=n, random_state = 1)
#    rf.fit(X_train, y_train)
#    y_pred = rf.predict_proba(X_val)[:, 1]
#    auc = roc_auc_score(y_val, y_pred)
#    scores.append((n,auc))
#df_scores = pd.DataFrame(scores, columns=['n_estimators','auc'])
#plt.plot(df_scores.n_estimators, df_scores.auc)
#plt.show()

# it stops growning at a moment so the ideal would be to stop
# lets tune other paramenters


#scores = []
#for d in [5, 10, 15]:
#    for n in range(10, 201, 10):
#        rf = RandomForestClassifier(n_estimators=n, max_depth=d, random_state = 1)
#        rf.fit(X_train, y_train)
#        y_pred = rf.predict_proba(X_val)[:, 1]
#        auc = roc_auc_score(y_val, y_pred)
#        scores.append((d, n,auc))
#df_scores = pd.DataFrame(scores, columns=['max_depth','n_estimators','auc'])
#for d in [5, 10, 15]:
#    df_subset = df_scores[df_scores.max_depth == d]
#    plt.plot(df_subset.n_estimators, df_subset.auc,label='max_depth = %s' % d)
#plt.legend()
#plt.show()

# best one looks like max_depth 10, so it does matter in random forest
# now lets do the same for min_leaf_sample

max_depth = 10
#scores = []
#
#for s in [1,3, 5, 10, 55]:
#    for n in range(10, 201, 10):
#        rf = RandomForestClassifier(n_estimators=n, max_depth=max_depth,min_samples_leaf=s, random_state = 1)
#        rf.fit(X_train, y_train)
#        y_pred = rf.predict_proba(X_val)[:, 1]
#        auc = roc_auc_score(y_val, y_pred)
#        scores.append((s, n, auc))
#df_scores = pd.DataFrame(scores, columns=['min_samples_leaf','n_estimators','auc'])
#for s in [1, 3, 5, 10, 55]:
#    df_subset = df_scores[df_scores.min_samples_leaf == s]
#    plt.plot(df_subset.n_estimators, df_subset.auc,label='min_samples_leaf = %s' % s)
#plt.legend()
#plt.show()

# 50 and 10 are worse than 1 3 and 5. Around 100 estimators, it plateaus.
min_samples_leaf = 3
# lets train with best values
#rf = RandomForestClassifier(n_estimators=n, max_depth=max_depth,min_samples_leaf=min_samples_leaf, random_state = 1)
#rf.fit(X_train, y_train)

# we can also train with parameter "max_features" to be different, which is how many features are selected in each model.
# Bootstrap is also another nice parameter

# all of these models can be trained in parallel, so we can use n_jobs to parallelize the training.

rf = RandomForestClassifier(n_estimators=100, max_depth=max_depth,min_samples_leaf=min_samples_leaf, random_state = 1, n_jobs=-1) #n_jobs -1 takes all cores
rf.fit(X_train, y_train)
y_pred = rf.predict_proba(X_val)[:, 1]
print(roc_auc_score(y_val, y_pred))
    
print( "6.7 Gradient boosting and XGBoost ") 
print ()
# this is another way of combining all models/decision trees instead of random forest
# in random forest, each of them is independet from the others
# when we train sequenctially, each model that we train after can fix the errors from the previous one.
# this way of combining models it is called boosting

# pip3 install xgboost

import xgboost as xgb

features = dv.get_feature_names()
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)
xgb_params = {
        'eta': 0.3, #how fast it trains
        'max_depth': 6,
        'min_child_weight': 1,  #te same as min_samples_leaf
        'objective': 'binary:logistic', #becasue we have a binary class type
        'nthread': 8,
        'seed': 1,
        'verbosity':1,
        }
model = xgb.train(xgb_params, dtrain, num_boost_round=10)
y_pred = model.predict(dval)
auc = roc_auc_score(y_val, y_pred)

print(auc) #for default parameters we get 0.80, pretty good

# we can evaluate the mode after each training since it is sequential

watchlist = [(dtrain, 'train'),(dval,'dval')]
xgb_params = {
        'eta': 0.3, #how fast it trains
        'max_depth': 6,
        'min_child_weight': 1,  #te same as min_samples_leaf
        'objective': 'binary:logistic', #becasue we have a binary class type
        'eval_metric':'auc',
        'nthread': 8,
        'seed': 1,
        'verbosity':1,
        }
model = xgb.train(xgb_params, dtrain, num_boost_round=200,evals=watchlist,verbose_eval=5)
# we see that in the train dataset we get a 1, and it stagenates after 100 steps, so it overfits at around 100

# it is difficult to get the output of the eval because it prints internally.
# in jupyter notebook we can use %%capture output and then output.stdout will give the text, but in normal python it is not so easy

#I found my own way:

from io import StringIO 
import sys

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout
with Capturing() as output:
    model = xgb.train(xgb_params, dtrain, num_boost_round=200,evals=watchlist,verbose_eval=5)


def parse_xgb_output(output):
    output = list(output)
    results = []

    #for line in output.strip().split(','):
    for line in output:
        it_line, train_line, val_line = line.split('\t')

        it = int(it_line.strip('[]'))
        train = float(train_line.split(':')[1])
        val = float(val_line.split(':')[1])

        results.append((it,train,val))

    columns = ['num_iter','train_auc','val_auc']
    df_results = pd.DataFrame(results,columns=columns)
    return df_results

df_score = parse_xgb_output(output)
#plt.plot(df_score.num_iter,df_score.train_auc, label='train')
#plt.plot(df_score.num_iter,df_score.val_auc, label='val')
#plt.show()


print( "6.8 XGBoost parameter tuning ") 
print ()

# we will tune eta, max_depth and min_child_weight
# eta is the learning rate, it says how much weight it has the second model when the first model is corrected
# if we use 1, all the errors are used to correct.
# it is also called size of step

scores = {}
for etas in [0.3, 1.0, 0.1, 0.05, 0.01]:
    xgb_params = {
            'eta': etas, #how fast it trains
            'max_depth': 6,
            'min_child_weight': 1,  #te same as min_samples_leaf
            'objective': 'binary:logistic', #becasue we have a binary class type
            'eval_metric':'auc',
            'nthread': 8,
            'seed': 1,
            'verbosity':1,
            }
    with Capturing() as output:
        model = xgb.train(xgb_params, dtrain, num_boost_round=200,evals=watchlist,verbose_eval=5)
    key = 'eta=%s' % (xgb_params['eta'])
    scores[key] = parse_xgb_output(output)

#for key, df_score in scores.items():
#    plt.plot(df_score.num_iter, df_score.val_auc, label=key)
#plt.legend()
#plt.show()

# we get different values compared to the teacher, but we will stick to eta = 0.1
# he first tunes eta, then max_depth and then min_child_weight

scores = {}
for d in [6, 3, 4, 10]:
    xgb_params = {
            'eta': 0.1, #how fast it trains
            'max_depth': d,
            'min_child_weight': 1,  #te same as min_samples_leaf
            'objective': 'binary:logistic', #becasue we have a binary class type
            'eval_metric':'auc',
            'nthread': 8,
            'seed': 1,
            'verbosity':1,
            }
    with Capturing() as output:
        model = xgb.train(xgb_params, dtrain, num_boost_round=200,evals=watchlist,verbose_eval=5)
    key = 'max_depth=%s' % (xgb_params['max_depth'])
    scores[key] = parse_xgb_output(output)

#for key, df_score in scores.items():
#    plt.plot(df_score.num_iter, df_score.val_auc, label=key)
#plt.legend()
#plt.show()

# for max depth looks like 3 is best, and it plateaus at 175 more or less

scores = {}
for m in [1, 10, 30]:
    xgb_params = {
            'eta': 0.1, #how fast it trains
            'max_depth': 3,
            'min_child_weight': m,  #te same as min_samples_leaf
            'objective': 'binary:logistic', #becasue we have a binary class type
            'eval_metric':'auc',
            'nthread': 8,
            'seed': 1,
            'verbosity':1,
            }
    with Capturing() as output:
        model = xgb.train(xgb_params, dtrain, num_boost_round=200,evals=watchlist,verbose_eval=5)
    key = 'min_child_weight=%s' % (xgb_params['min_child_weight'])
    scores[key] = parse_xgb_output(output)

#for key, df_score in scores.items():
#    plt.plot(df_score.num_iter, df_score.val_auc, label=key)
#plt.legend()
#plt.show()

# we will go with 30, also with 175 iterations
scores = {}
xgb_params = {
        'eta': 0.1, #how fast it trains
        'max_depth': 3,
        'min_child_weight': 30,  #te same as min_samples_leaf
        'objective': 'binary:logistic', #becasue we have a binary class type
        'eval_metric':'auc',
        'nthread': 8,
        'seed': 1,
        'verbosity':1,
        }

model = xgb.train(xgb_params, dtrain, num_boost_round=175)

# other important other useful parameters are subsample and colsample_bytree


print( "6.9 Selecting the final model") 
print ()

# we trained a decision tree, a random forest and a gradient boost. 
# we want to evaluate which is the best
dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf = 15)
dt.fit(X_train, y_train)
y_pred = dt.predict_proba(X_val)[:, 1]
print ("validation: " ,roc_auc_score(y_val, y_pred)) 

rf = RandomForestClassifier(n_estimators=200, max_depth=10,min_samples_leaf=3, random_state = 1, n_jobs=-1) #n_jobs -1 takes all cores
rf.fit(X_train, y_train)
y_pred = rf.predict_proba(X_val)[:, 1]
print ("validation: " ,roc_auc_score(y_val, y_pred)) 

xgb_params = {
        'eta': 0.1, #how fast it trains
        'max_depth': 3,
        'min_child_weight': 30,  #te same as min_samples_leaf
        'objective': 'binary:logistic', #becasue we have a binary class type
        'eval_metric':'auc',
        'nthread': 8,
        'seed': 1,
        'verbosity':1,
        }

model = xgb.train(xgb_params, dtrain, num_boost_round=175)
y_pred = model.predict(dval)
auc = roc_auc_score(y_val, y_pred)
print ("validation: {}".format(auc))


print ("Training the gradient boost in the full dataset")
df_full_train = df_full_train.reset_index(drop=True)
y_full_train = (df_full_train.status == "default").astype(int).values
del df_full_train["status"]
dicts_full_train = df_full_train.to_dict(orient="records")
dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)
dicts_test = df_test.to_dict(orient="records")
X_test = dv.transform(dicts_test)
dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=dv.get_feature_names())
dtest = xgb.DMatrix(X_test, feature_names=dv.get_feature_names())
model = xgb.train(xgb_params, dfulltrain, num_boost_round=175)
y_pred = model.predict(dtest)
auc = roc_auc_score(y_test, y_pred)
print ("validation: {}".format(auc))
