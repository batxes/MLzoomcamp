#  Capstone project: Map type classification![honolulu](https://github.com/batxes/MLzoomcamp/blob/main/capstone1/image.jpg)

## About the project

This is a Machine Learning project that aims to predict or classify satellite pictures into classes of lands.  
I downloaded the data from: https://www.tensorflow.org/datasets/catalog/eurosat#eurosatrgb_default_config
This dataset contains 27000 labeled pictures and they consist in total 10 different classes of lands.
I am very interested in **climate change** and I think that a deep learning model predicting satellite images may be the very basic if I want to classify or predict other kind of satellite images in the future.  

## Description of the problem

The problem consists on classifying satellite images into classes of lands. There are 10 classes of lands in total:
1. "AnnualCrop"
1. "Forest"
1. "HerbaceousVegetation"
1. "Highway"
1. "Industrial"
1. "Pasture"
1. "PermanentCrop"
1. "Residential"
1. "River"
1. "SeaLake"

To tackle the problem, we will create a deep learning model.

## About the Model

The model is convolutional neural network.

## Repository contains

 - **Readme.md**: with description of the problem and instructions on how to run the project
 - **EuroSAT_RGB**: Directory with all the images.
 - **notebook.ipynb**: python notebook with:
	 - Data preparation and data cleaning
	 - EDA and feature importance analysis
	 - Model selection process and parameter tuning
     - Model saving
 - **classify.py**: script that loads the model and serves it via a web service with **flask**
 - **test.py**: script used to test the model. It loads a image and tests.
 - **Pipfile** and **Pipfile.lock**: files with the library dependencies
 - **Dockerfile** with the instructions to build the docker image
 - **final-map-model.h5** final model

## How to run the model

Model can be run locally in these ways:
 1. locally:
    1. `pipenv install flask gunicorn tensorflow pillow`
    2. `pipenv shell`
    3. `gunicorn --bind 0.0.0.0:9696 classify:app` 
    4. in another terminal: `python3 test.py`
 2. Docker:
    1. `docker build -t ml-zoomcamp-maps .`
    2. `docker run -p 9696:9696 -it --rm  ml-zoomcamp-maps`
    3. in another terminal: `python3 test.py`

### Instructions and code of the work carried out
 1. I first created a notebook called **notebook.ipynb** where I downloaded the data, explored, prepared the data, run models, tunned the models with different paramenters, evaluated them and concluded which was the model that performed the best. The model then was saved.
 2.  Then I created and environment and installed the libraries I will be using: `pipenv install flask gunicorn tensorflow pillow`
 3. Then I run the environment with: `pipenv shell`
 4. I uploaded one of the images to imgur: https://i.imgur.com/6n71Nae.jpg
 5. I created `test.py` and `classify.py`
 6. Then I tested the model. Run `classify.py` in the environment and test.py in another terminal. Works.
 7. Next step is to containerize with Docker. I created a Dockerfile and build the container with: `docker build -t ml-zoomcamp-maps .`
 8. Run the container with: `docker run -p 9696:9696 -it --rm  ml-zoomcamp-maps` and test running `python3 test.py` 

### Improvements

 I did not have the time but these would be the next improvements:
  1. Convert the model to tf-lite and load that in the docker container, so we do not download all the tf library, which is huge
  2. I would try to check other models that are already created and use transfer learning.
  3. Deploy the containers with kubernertes.


