# in previous session we trained a model to classify images with keras. 
# Now we will deploy this model. One way is using AWS Lambda. 
# We send the URL to lambda, and then this service will reply many classes of clothes.
# FOr this, we will use TF-lite

# 9.2. What is AWS Lambda? In the website says: Run Code without thinking about servers.
# SO lambda takes care of everything, not like Elastic Beanstalk

# https://eu-north-1.console.aws.amazon.com/lambda/home?region=eu-north-1#/functions
# basically, we write the code in the lamda function and that's it, we do not need to do anything else.

# this is convinient because we do not need to worry about servers and maybe during the day we have traffic  and at night no, so we save money also

# for this we do not need to use TF because it is too big, so we will use tf lite

# 9.3 TensorFlow Lite
# it is much lighter than the normal and is convinient because there are limits in some servers.
# also, large image is problematic because it is more expemnsive to store and also takes more time to initialize.
# and also the import tf is larger than the lite

# TF lite focuses on inference. WHich is model.predict(X). That is inference. It is not used to training model, only model.predict. It can only do that.

# First, we will download https://github.com/DataTalksClub/machine-learning-zoomcamp/releases/download/chapter7-model/xception_v4_large_08_0.894.h5
# this is already trained by Alexei.





# we have a notebook with all the cod from tflite.
# we want to deply that to lambda

#use jupyter nbconvert --to script '09-serverless-tensorflow-model.ipynb'
# to convert the notebokk to script

# change the name of the script to lambda_function.py

