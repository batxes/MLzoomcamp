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
 - **train.py**: script that trains the model and saves it to a model with **pickle**
 - **predict.py**: script that loads the model and serves it via a web service with **flask**
 - **test.py**: script that contains a possible day that is used to test the model and predict the next day.
 - **Pipfile** and **Pipfile.lock**: files with the library dependencies
 - **Dockerfile** with the instructions to build the docker image
 

## Instructions on how to run the project
I recorded a video on how to run the project. 
### Video

[![Instructions video](https://img.youtube.com/vi/rLU9D3jbrng/maxresdefault.jpg)](https://youtu.be/rLU9D3jbrng)



### Instructions and code of the work carried out


 1. I first created a notebook called **notebook.ipynb** where I downloaded the data, explored, prepared the data, cleaned, run different models with different parameters, evaluated them and concluded which was the model that performed the best. The model then is converted into tf-lite and saved.
 2.  Then I created and environment and installed the libraries I will be using: `pipenv install flask gunicorn tensorflow pillow`
 3. Then I run the environment with: `pipenv shell`
 4. I uploaded one of the images to imgur: https://i.imgur.com/6n71Nae.jpg
 5. I created test.py and classify.py
 6. Then I tested the model. Run classify.py in the environment and test.py in another terminal. Works.
 7. Next step is to containerize with Docker. I created a Dockerfile and build the container with: docker build -t ml-zoomcamp-maps .
 8. Run the container with: `docker run -it --rm  ml-zoomcamp-maps` and test running python3 test.py 


 7. After checking that it works, I create a **docker** container `sudo docker build -t midterm_project .`. We can test it running the docker image `docker run -it --rm  midterm_project` and executing `python3 test.py`
 8. Finally I deploy it to **AWS** with **Elastic Beanstalk**. For that I first install the library `pipenv install awsebcli --dev`, initialize EB `eb init -p docker -r eu-north-1 midterm_project` and create the service `eb create midterm-project-env`
 9. Now we just need to test it. For that, I modified the line pointing to the url in `test.py` and run `python3 test.py`. (There is no need to change it now since the server is still running in AWS (today being 5 November 2023))

### Miscelanea
I had some problems with getting the feature names from the DictVectorizer.  
In the course we used `dv.get_feature_names()` but I got errors on my end and I had to change to `list(dv.get_feature_names_out())`.  
After reading it looks like it has to do with different scykit versions.  
I also read that `get_feature_names()` is being replaced to `get_feature_names_out()` in the library so I kept it like that.  


