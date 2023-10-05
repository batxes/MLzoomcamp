# Oct 25

#### 4. Evaluation Metrics for Classification

print ("\n\n\n###")
print ("4.1 Evaluation metrics: session overview\n")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('../03-classification/WA_Fn-UseC_-Telco-Customer-Churn.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values

del df_train['churn']
del df_val['churn']
del df_test['churn']

numerical = ['tenure', 'monthlycharges', 'totalcharges']

categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]

dv = DictVectorizer(sparse=False)

train_dict = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

model = LogisticRegression()
model.fit(X_train, y_train)

val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)

y_pred = model.predict_proba(X_val)[:, 1]
churn_decision = (y_pred >= 0.5)
print ((y_val == churn_decision).mean())

print ("\n\n\n###")
print ("4.2 Accuracy and dummy model\n")

print ("Number of customers: {}".format(len(y_val)))
print ("Number oc correct decissions: {}".format((y_val == churn_decision).sum()))
print ("Accuracy is: correct predictions / total number of customers.")
print ("We can divide both or just get the mean of y_val == churn decission")

print ()

print ("BUt this comes from using a decision cutoff of 0.5 for churning. We can change this one actually and see if it improves our accuracy.")
thresholds = np.linspace(0, 1, 21) #we can treat each of this as cutoff

scores = []

for t in thresholds:
    churn_decision = (y_pred >= t)
    #score = (y_val == churn_decision).mean()
    score = accuracy_score(y_val, y_pred >= t)
    print ('%.2f %3f' % (t, score))
    scores.append(score)

#plt.plot(thresholds,scores)
#plt.show()
print ("we see that 0.5 actually has the best score")

#we can use accuracy score from scikit instead of getting our own accuracy
print ("we can get the accuracy with scikit also")
print (accuracy_score(y_val, y_pred >= 0.5))

# customers are not churning if score is above 1 and opposite if it is 0, so limits
from collections import Counter
print (Counter(y_pred >= 1))
print ("If we use a dummy model that says all are not churning we get a accuracy of 72%")
print ("Dummy model is 72% accurate and our model 80%. SO why bother?")
print ("The thing is that we have an imbalance of non churning people and churning. So we need a Dummy model that is 50/50.")
print ("Accuracy is misleading for the imbalance cases so we will learn how to evaluate them.")


print ("\n\n\n###")
print ("4.3 Counfusion table\n")

print ("Way of looking at different errors and correct decisions of our classification model.")

print ("We will count True/False negatives/positives")

actual_positive = (y_val == 1)
actual_negative = (y_val == 0)
t = 0.5
predict_positive = (y_pred >= t)
predict_negative = (y_pred < t)

print (predict_positive[:5])
print (actual_positive[:5])
print ((predict_positive & actual_positive)[:5]) 

tp = (predict_positive & actual_positive).sum() #this return a list where is True in both
tn = (predict_negative & actual_negative).sum() #this return a list where is True in both also, but this will be true negatives
fp = (predict_positive & actual_negative).sum()
fn = (predict_negative & actual_positive).sum()

print ("get confussion matrix")
confusion_matrix = np.array([
    [tn, fp],
    [fn, tp]
])
print (confusion_matrix)
print ((confusion_matrix/confusion_matrix.sum()).round(2))


print ("\n\n\n###")
print ("4.4 Precision and Recall\n")
print ("Accuracy is also: (tp+tn) / (tp+tn+fp+fn))")
print ((tp+tn) / (tp+tn+fp+fn))
print ("Precision is how many ppositive predictions are correct (or fraction of positive predictions that are correct)")
print ("TP/# of positives --> TP/(TP+FP)")
p = tp/(tp+fp)
print (p)

print ("Recall in the other hand is the fraction of correclty identified positive examples")
print ("TP/#positive observations --> TP/(TP+FN)")

r= tp/(tp+fn)
print (r)

print ("So using accuracy in this models is not good, it says 80%, but it is better to check recall, where we can see that we failed to identify 46% of users (100-recall) and we sent email with discount to 33% of people (100-67%) from false positives")
print ("That is why are useful metrics for imbalanced data specially.")

print ("\n\n\n###")
print ("4.5 ROC curves\n")

print ("For roc curves we want FPR (False postiive rate) and TPR (True positive rate)")
print ("FPR = FP/(TN+FP)")
print ("TPR = TP/(FN+TP)")

print ("We want as much as TPR and as less as FPR")
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)
print (tpr, fpr)
print ("ROC curve evaluates these cualities for all thresholds")

thresholds = np.linspace(0, 1, 21) #we can treat each of this as cutoff

scores = []

for t in thresholds:
    actual_positive = (y_val == 1)
    actual_negative = (y_val == 0)
    predict_positive = (y_pred >= t)
    predict_negative = (y_pred < t)
    tp = (predict_positive & actual_positive).sum() #this return a list where is True in both
    tn = (predict_negative & actual_negative).sum() #this return a list where is True in both also, but this will be true negatives
    fp = (predict_positive & actual_negative).sum()
    fn = (predict_negative & actual_positive).sum()
    scores.append((t,tp,fp,tn,fn))
columns = ['threshold','tp','fp','tn','fn']
df_scores = pd.DataFrame(scores,columns=columns)
df_scores["tpr"] = df_scores.tp / (df_scores.tp + df_scores.fn)
df_scores["fpr"] = df_scores.fp / (df_scores.fp + df_scores.tn)
print (df_scores)
#plt.plot(df_scores.threshold,df_scores['tpr'],label='TPR')
#plt.plot(df_scores.threshold,df_scores['fpr'],label='FPR')
#plt.legend()
#plt.show()
print ("In this plot, we want to minimize FPR as fast as posible and TPR needs to be kept around 1 all the time. Our threshold is 0.5")
print ("Now we want to plot a random model to compare, a dummy model.")

np.random.seed(1)
y_rand = np.random.uniform(0,1,size=len(y_val))
print ("Accuracy for this model: {}".format(((y_rand >= 0.5) == y_val).mean()))

def tpr_fpr_dataframe(y_val, y_pred):
    thresholds = np.linspace(0, 1, 21) #we can treat each of this as cutoff

    scores = []

    for t in thresholds:
        actual_positive = (y_val == 1)
        actual_negative = (y_val == 0)
        predict_positive = (y_pred >= t)
        predict_negative = (y_pred < t)
        tp = (predict_positive & actual_positive).sum() #this return a list where is True in both
        tn = (predict_negative & actual_negative).sum() #this return a list where is True in both also, but this will be true negatives
        fp = (predict_positive & actual_negative).sum()
        fn = (predict_negative & actual_positive).sum()
        scores.append((t,tp,fp,tn,fn))
    columns = ['threshold','tp','fp','tn','fn']
    df_scores = pd.DataFrame(scores,columns=columns)
    df_scores["tpr"] = df_scores.tp / (df_scores.tp + df_scores.fn)
    df_scores["fpr"] = df_scores.fp / (df_scores.fp + df_scores.tn)
    return df_scores

df_rand = tpr_fpr_dataframe(y_val, y_rand)
print (df_rand)

#plt.plot(df_rand.threshold,df_rand['tpr'],label='TPR')
#plt.plot(df_rand.threshold,df_rand['fpr'],label='FPR')
#plt.legend()
#plt.show()

print ("This plot is linear, pretty bad of course.")

print ("Lets tak about a different benchmark, the ideal model")
print ("We will create an array of zeros first and then only ones")

num_neg = (y_val == 0).sum()
num_pos = (y_val == 1).sum()
print (num_neg, num_pos)
print ("We can put the cutoff in 72% to get all of them right")
print (1-y_val.mean())
y_ideal = np.repeat([0,1],[num_neg, num_pos])
print (y_ideal)

y_ideal_pred = np.linspace(0,1,len(y_val))
print ("accuracy = {}".format(((y_ideal_pred >= 0.726) == y_ideal).mean()))

#nowe wd do a dataframe for tjhe ieal model

df_ideal = tpr_fpr_dataframe(y_ideal,y_ideal_pred)
#plt.plot(df_ideal.threshold,df_ideal['tpr'],label='TPR')
#plt.plot(df_ideal.threshold,df_ideal['fpr'],label='FPR')
#plt.legend()
#plt.show()

print ("this model can identify correctly all if we use 72% as cutoff. We make mistakes if we use other cutoffs")

print ("we can plot all together")

#plt.plot(df_scores.threshold,df_scores['tpr'],label='TPR')
#plt.plot(df_scores.threshold,df_scores['fpr'],label='FPR')
#plt.plot(df_ideal.threshold,df_ideal['tpr'],label='TPR',color="black")
#plt.plot(df_ideal.threshold,df_ideal['fpr'],label='FPR',color="black")
#plt.legend()
#plt.show()

print ("we want our model to be as close as the ideal")
print ("we plot ROC curves for tthat, plotting FPR against TPR")

#plt.figure(figsize=(6,6))
#plt.plot(df_scores.fpr, df_scores.tpr,label="model")
##plt.plot(df_rand.fpr, df_rand.tpr,label="random") #actually for the random we normally plot a line
#plt.plot([0,1],[0,1],label="random")
##plt.plot(df_ideal.fpr, df_ideal.tpr,label="ideal") #and we do not need to plot ideal because we know is an "L"
#plt.legend()
#plt.show()

print ("In this curve, if we use a cutoff of 0.7, we start predicting some and our TPR is starting to increase but also a little bit the FPR, but later when we use lower cutoffs, we do not predict so much well and we make more mistakes.")

print ("we can use roc_curve function from scikit")
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_val, y_pred)
#plt.figure(figsize=(6,6))
#plt.plot(fpr,tpr,label="model")
#plt.plot([0,1],[0,1],label="random")
#plt.legend()
#plt.show()

#this one is more shaky because the function uses more cutoffs, so it is more accurate

print ("Now we will work with the area under the roc curve: ROC AUC. Good metric to evaluate binary classification models")

print ("for a random model the area is half of the square, so 0.5")
print ("for the ideal is AUC = 1")
print ("So, 0.8 is good, 0.9 would be great and 0.6 so so")

from sklearn.metrics import auc

print (auc(fpr, tpr))
print (auc(df_scores.fpr, df_scores.tpr))
print ("The one from scikit is better for evaluating because it uses more thresholds.")

from sklearn.metrics import roc_auc_score

print ("We can do a shortcut directly using the actual values and prediction and computes the score directly")
print (roc_auc_score(y_val, y_pred))

print ("AUC has a very good intepretation: AUC = probability that randomly selected positive has higher score than randomly selected negative ")
print ()
print ()
print ("4.7: Cross validation")

print ("Evaluating the same model on different subsets of data")
print ("what we do is take different chunks of training data, and we use a part of the training for validation and the rest for training, but we repeat this process using different chunks")
print ("Whenever we do that, we calculate the AUC, and then the mean of the AUC and the std deviation")

def train (df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C,max_iter = 1000)
    model.fit(X_train, y_train)

    return dv, model

dv, model = train (df_train, y_train,C=0.001) #lower values, stronger regularization

def predict (df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')
    
    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

y_pred = predict(df_val, dv, model)

from sklearn.model_selection import KFold

kfold = KFold(n_splits=10, shuffle=True, random_state=1)
print (next(kfold.split(df_full_train))) # we can use next to print what we have inside
train_idx, val_idx = next(kfold.split(df_full_train)) # we can use next to print what we have inside

df_train = df_full_train.iloc[train_idx]
df_val = df_full_train.iloc[val_idx]
print ("now we have train and val dataframes, so we can do the same as always, training the model. Instead of using next we will use a loop")

print ("to see the progress, we can install tqdm library")

from tqdm.auto import tqdm

n_splits = 5

for C in tqdm([ 0.001, 0.01, 0.1, 0.5, 1, 5, 10]):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

    scores=[]

    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]
        
        y_train = df_train.churn.values
        y_val = df_val.churn.values

        dv, model = train(df_train,y_train, C=C)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)

    print (C, np.mean(scores),np.std(scores))
    #print ("We talked about parameter tunning, and we can modify these in the logistic regression function as a parameter")
    #print ("We can add the parameter in our train function")

print ("Looks like C=1 is the best. SO using those parameteters, we use it to train the full train dataset")

dv, model = train (df_full_train, df_full_train.churn.values,C=1)
y_pred = predict(df_test, dv, model)
auc=roc_auc_score(y_test,y_pred)
print (auc)


