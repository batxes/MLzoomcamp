pipenv install awsebcli --dev
data: https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package

First we did the notebook

then we have the model and we wrote the train.py, predict.py and test.py


now in the folder we run:
pipenv install numpy scikit-learn flask gunicorn xgboost

pipenv shell
now we can put the service predict.py in this envyronment

run: gunicorn --bind 0.0.0.0:9696 predict:app

then run: python3 test.py

We see it works: 

{'rain_probability': 0.0008115128730423748, 'rain_tomorrow': False}
tomorrow will NOT rain. Very low probability 0.0008115128730423748.

now, create a Dockerfile

and build the image with: sudo docker build -t midterm .

# I will try in in laptop because docker does not connect to internet in my computer

then run again the test.py to see that I can reach the docker

then finally deploy in AWS.

pipenv install awsebcli --dev

eb init -p docker -r eu-north-1 midterm_project

eb create midterm-project-env

now, I need to change the fisrt lines of test.py and run it







