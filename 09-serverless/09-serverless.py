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

#9.4 PREPARING THE LAMBDA CODE.

# Check the notebook and the lambda.py

# 9.5 PREPARING A DOCKER IMAGE

# check the docker image
# to do the base image we go to public.ecr.aws. Take python lambda
# we will take python 3.8
# public.ecr.aws/lambda/python:3.8

# after creating the docker function we build it
# docker build -t clothing-model .

# I had a problem with docker connection to HTTPS: I managed to do this and worked;

#{
#  "dns": ["8.8.8.8", "8.8.4.4"]
#}
#and restart the docker service:
#
#sudo service docker restart

# the error was:
#WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x7fd33c20fe50>: Failed to establish a new connection: [Errno -3] Temporary failure in name resolution')': /simple/keras-image-helper/

# we do not need to install in the docker image numpy and other libraries because tflite already depends on them

# then run it:
# docker run -it --rm -p 8080:8080 clothing-model:latest

# create test.py to test it while running in docker

# when testing there was an error, 
# {'errorMessage': "Unable to import module 'lambda_function': /lib64/libm.so.6: version `GLIBC_2.27' not found (required by /var/lang/lib/python3.8/site-packages/tflite_runtime/_pywrap_tensorflow_interpreter_wrapper.so)", 'errorType': 'Runtime.ImportModuleError', 'stackTrace': []}

# this is because the librarye thats get installed in docker (tflite_runtime) is compiled in ubuntu while in lambda we use amazon_linux (centos based)
# we need to compile now tflite_runtime in amazon_linux.
# Alexei did that already: https://github.com/alexeygrigorev/tflite-aws-lambda/tree/main/tflite
# we can also create our own. He has instructions for that
# we will use: https://github.com/alexeygrigorev/tflite-aws-lambda/blob/main/tflite/tflite_runtime-2.7.0-cp38-cp38-linux_x86_64.whl

# actually I had problems with version so I will work with python3.1 and tflite_runtime 2.14
# in the end, I had to download the whl file and copy to the docker file, without directly downloading

# now we get another error: {'errorMessage': 'Unable to marshal response: Object of type float32 is not JSON serializable', 'errorType': 'Runtime.MarshalError', 'requestId': '764e9896-a9f4-4288-948e-3ddb2586fd65', 'stackTrace': []}

# we had the same error with flask in the prevuious sessions
# we will do a shortcut modifying lambad_function.py to get python list 

#now build and run again;
#
#docker build --platform linux/amd64 -t clothing-model .
#docker run -it --rm -p 8080:8080 clothing-model:latest

# 9.6 Creating the lambda function

# now, instead of creating from scrath the lambda function, we will select container image in the AWS web
# we need to upload our docker image to the cloud. We can do it with amazon ECR

# we can do the creation of a repository with command line
# pip install awscli
# aws ecr create-repository --repository-name clothing-tflite-images

# first we do aws configure: I aded access key, secret key(both in google keep) and eu-north-1
# then copy repository uria into the website AES where it asked the URI
# but before that we have to register and also tell that we will be using passweords
# aws ecr get-login --no-include-email | sed 's/[0-9a-zA-Z=]\{20,\}/PASSWORD/g'
# we run the output of this. To do that, run like this:
#$(aws ecr get-login --no-include-email) # this runs the output of whatever is inside brackets

# lets run the next code:
ACCOUNT=302181200248
REGION=eu-north-1
REGISTRY=clothing-tflite-images
PREFIX=${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REGISTRY}

TAG=clothing-model-xception-v4-001
REMOTE_URI=${PREFIX}:${TAG}

# we will push the remote_uri
# first tag:
docker tag clothing-model:latest ${REMOTE_URI}
# then push
docker push ${REMOTE_URI}
# this will pish the image to ECR AWS
# lets check here: https://eu-north-1.console.aws.amazon.com/ecr/repositories?region=eu-north-1
# under images

# now we can use the docker image in lambda: https://eu-north-1.console.aws.amazon.com/lambda/home?region=eu-north-1#/create/function?intent=authorFromImage

#the uri is: 302181200248.dkr.ecr.eu-north-1.amazonaws.com/clothing-tflite-images:clothing-model-xception-v4-001
#now lets test it. Go to test. put this code:
{"url":"http://bit.ly/mlbookcamp-pants"}

# it fails because default timeout is 3 seconds. CHange that in configuration to 30 and 1024 mb memory
# we can also check how much we pay for each request.

# we pay more or less 33 cents to classify 1000 images

# 9.7 API GATEWAY: EXPOSING LAMBDA FUNCTION

#go to API gateway in aws.
# create a new one.
# select rest API
# after creating, we want to create a resource
# name: predict
# for the resource, we create a method POST
# then, in the website, we test the method.

# now we take the API gateway and deploy it. For that we need to expose it.

# now we have a URL to invoke: https://y8dpr8i7l6.execute-api.eu-north-1.amazonaws.com/test
# put it in the test.py url with /predict in the end  and run it.



