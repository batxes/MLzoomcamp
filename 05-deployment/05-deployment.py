

# 5.2 Saving and loading the model
#  - Saving the model to pickle
#  - Loading the model from Pickle
#  - Turning our notebook into python script

# this is the model from week 3

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

df = pd.read_csv('data-week-3.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

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

def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)

    return dv, model

def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred
C = 1.0
n_splits = 5

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))

dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)

y_test = df_test.churn.values

auc = roc_auc_score(y_test, y_pred)
print (auc)

# pip3 install pickle-mixin

# pickle will be used to save the model

import pickle

output_file = f'model_C={C}.bin'
print(output_file)

f_out = open(output_file,'wb')
pickle.dump((dv,model),f_out) # we also need the dict vectorizer to understand later the prediction, not only the model. SO we put both in a tuple
f_out.close()

#this can also be done
#with open(output_file, 'wb') as f_out: 
#    pickle.dump((dv, model), f_out)

#to load the model:

import pickle

with open(output_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}
X = dv.transform([customer]) #convert the customer into feature matrix
y_pred = model.predict_proba(X)[0, 1] #use the model to predict. We are interested in the second value, the probability

print('input:', customer)
print('output:', y_pred)

################   5.3. Web services. Introduction to Flask

# we first craete a file called ping.py
# pip3 install flask

#that is a web service. We can test it with: curl http://0.0.0.0:9696/ping


################   5.4. Serving the Churn with Flask
# we will create the churning web service now

# we called it predict.py. 

# we also need to have predict.py running.

# then, we will send a POST request

#making requests

import requests
url = 'http://localhost:9696/predict'

customer={
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "no",
    "phoneservice": "no",
    "multiplelines": "no_phone_service",
    "internetservice": "dsl",
    "onlinesecurity": "no",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "two_year",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "tenure": 1,
    "monthlycharges": 29.85,
    "totalcharges": 29.85
}

response = requests.post(url, json=customer).json()

print (response)

if response['churn']:
    print('sending email to', 'asdx-123d')

# when we deploy predict.py, it says: WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.

# we can install and use gunicorn for that

# pip3 install gunicorn

# then run with: gunicorn --bind 0.0.0.0:9696 predict:app
# gunicorn only works with UNIX systems

# for windows we can use waitress

################   5.5. Python virtual environment: Pipenv

# if we want to install an specific package we should do envyronments
# maybe we want sckit learn 1.0 in one project but 2.0 in another
# keep separated because if we update a library it could be that a service doe snot work anymore
# use virtual envyronments for that

# pip3 install pipenv
# to install libraries in this a envyronment:

# pipenv install numpy scikit-learn==0.24.2 flask

# we got error with the scikit version, so we removed it and run again

# we can also run now, pipenv install gunicorn 
# this will rewrite the pip file

# now we want to run our service and get in the environment we created
# pipenv shell

#now, we we write which python3, we can see it goes to 
# /home/ibai/.local/share/virtualenvs/05-deployment-C2e2Iq9_/bin/python3

# now we can put the service predict.py in this envyronment 
# gunicorn --bind 0.0.0.0:9696 predict:app
# and run test.py



################   5.6. Environment Management: Docker

# check here all the images in python: https://hub.docker.com/_/python/
# we will use the python:3.8.12-slim
# docker run -it --rm python:3.8.12-slim

#it for using terminal in the image
#rm for removing after exiting

# we can use the terminal of the image overwriting it like this:
#docker run -it --rm --entrypoint=bash python:3.8.12-slim

#dockerfile
# creates a image

#check Dockerfile that I created
# to build the image: sudo docker build -t zoomcamp-test .

# now, we can run the image like this: docker run -it --rm --entrypoint=bash zoomcamp-test 

# and inside the image, we can run pipenv install to start installing our libraries needed for the web service

# we modified Dockerfile more, addint the lines to copy the model and the predict file to the docker image

# now, run Docker and we can execute the server: gunicorn --bind=0.0.0.0:9696 predict:app

# we can not access that port because we need to expose it first, 

# we need to map the port in the docker to the port in the host machine. Thenm our test.py will access the port in the host machine which will be mapped to the docker machine

# we do that also in the docker container, using the dockerfile and rebuilding the image
# we also want the docker machine to run the server directly, as entrypoint. SO we can write in the dockerfile: ENTRYPOINT gunicorn ...
# and then run the docker machine like this:
# sudo docker run -it --rm  zoomcamp-test

# if we now run test.py it will say it can not connect. That is because even though we exposed the 9696 port, we did not map it.
# to do that add to the docker run -p 9696:9696   first port is the container, and second port is the host
# sudo docker run -it --rm -p 9696:9696 zoomcamp-test


#########################  5.7 - Deployment To The Cloud: AWS Elastic Beanstalk

# create an account in AWS. Go to https://mlbookcamp.com/articles

# with Beanstalk runing inside AWS, we can have many instances of our service. We are scaling up in case we have lots of requests.  Beanstalk can also scale down in case the demand is low.# with Beanstalk runing inside AWS, we can have many instances of our service. We are scaling up in case we have lots of requests.  Beanstalk can also scale down in case the demand is low.

# for that, we install beanstalk, but we only want to install it developemtn:
# pipenv install awsebcli --dev

# then we initialize elastic beanstalk:
# eb init -p docker -r eu-north-1 churn-serving
# we want to run docker, in eu-north1 (berlin) and the name is churn serving.
# it will ask for ID and secret key. We can create in our AWS console: https://eu-north-1.console.aws.amazon.com/console/home?region=eu-north-1

# this creates a folder called .elasticbeanstalk and inside we have a yml file

# now we runt it :
#eb local run --port 9696 

# this does not work for me. Regardless, we will try to run on the cloud

# eb create churn-serving-env

# after a long time, it says that application is available at a certaina dress: churn-serving-env.eba-pwhtrpxp.eu-north-1.elasticbeanstalk.com

# now, execute test.py and we get our data. We can play with the customer to see that the value changes

#the problem with the host is that anybody in the world can access it if they know it. We should make sure that nobody gets it

# now lets terminate the server with:
# eb terminate churn-serving-env
