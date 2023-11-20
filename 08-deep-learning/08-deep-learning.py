import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# we can use sageMaker from AWS or googlecolab to do the exercises

# this week we will do a classifier of images. We will deploy a service and the user will upload images to a website that then will be fed to the service and it will upload what type of clothe it is from 10 different.

# 8.1: 

#   more info in: https://cs231n.github.io

# first clone the dataset : git clone https://github.com/alexeygrigorev/clothing-dataset-small.git

print ("\n8.2: TensorFlow and Keras\n")

# Tensorflow is a framework for doing/training deep learning models
# pip3 install tensorflow
# Keras is a higher level abstraction on top of TensorFlow

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.image import load_img
# for earlier versions is from keras.preprocessing.image import load_img

path = './clothing-dataset-small/train/t-shirt'
name = 'fcfc68be-41ea-40ef-8040-624453983a85.jpg'
fullname = f'{path}/{name}'
img = load_img(fullname, target_size = (150,150))
#print (img)

#each pixel is a cobination of red green and blue in each image, in a matrix of 150x150
x =  np.array(img)
#print (x)
#print (x.shape())

print ("\n8.3: Pre-trained convolutional neural networks\n")
# imagenet dataset: https://www.image-net.org/
# pre-trained models: https://keras.io/api/applications/










