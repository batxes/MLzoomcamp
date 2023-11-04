# Midtern project: Rain prediction![Kangaroos in the rain](https://github.com/batxes/MLzoomcamp/blob/main/midterm_project/kangaroos.jpg)

## Description of the problem

I am very interested in **climate change** and I thought that a model which **predicts the rain** could be interesting.
For this project I downloaded a Kaggle dataset which contains 10 years of daily weather observations in Australia.
There are many features like humidity, temperature and wind speed, but the target variable to predict will be **RainTomorrow**, which is associated to each day
and it states if it rained or not the next day. Training our model with all the features can help predict which type of day is the most likely that leads to
 a rainy tomorrow. I hope that this kind of projects can help me develop a more sophisticated model in the future to predict other kind of events related to weather and climate change. 

Link to the service: 
Dataset link: https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package


## Repository contains

 - **Readme.md**: with description of the problem and instructions on how to run the project
 - **weatherAUS.csv**: data used in the project.
 - **notebook.ipynb**: python notebook with:
	 - Data preparation and data cleaning
	 - EDA and feature importance analysis
	 - Model selection process and parameter tuning
 - **train.py**: script that trains the model and saves it to a model with **pickle**
 - **predict.py**: script that loads the model and serves it via a web service with **flask**
 - **test.py**: script that contains a possible day that is used to test the model and predict the next day.
 - **Pipfile** and **Pipfile.lock**: files with the library dependencies
 - **Dockerfile** with the instructions to build the docker image
 

## Instructions on how to run the project
I recorded a video on how to run the project. Regardless, below can be found the steps I took to carry on the project.
### Video
### Instructions

 1. I first created a notebook called **notebook.ipynb** where I downloaded the data, explored, prepared the data, cleaned, run different models with different parameters, evaluated them and concluded which was the model that performed the best.
 2. Then I generated the **train.py** script that trains the models and saves the model to a file with pickle, the **predict.py** file that loads the model and serves it and the **test.py** script that will be used to predict a specific day.
 3.  Then I created and environment and installed the libraries I will be using: `pipenv install numpy scikit-learn flask gunicorn xgboost`
 4. Then I run the environment with: `pipenv shell`
 5. I run a server locally: `gunicorn --bind 0.0.0.0:9696 predict:app`
 6. I test that the model is working with: `python3 test.py`
 7. After checking that it works, I create a **docker** container `sudo docker build -t midterm_project .` and test it again `python3 test.py`
 8. Finally I deploy it to **AWS** with **Elastic Beanstalk**. For that I first install the library `pipenv install awsebcli --dev`, initialize EB `eb init -p docker -r eu-north-1 midterm_project` and create the service `eb create midterm-project-env`
 9. Now we just need to test it. For that, we modify the line pointing to the url in `test.py` and run `python3 test.py`



