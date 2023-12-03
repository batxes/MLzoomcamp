

# 10.1

# we will use tensor flow serving. It is very efficient (C++) and is only focused on serving. We put the model in tf-serving and we just ask predictions.
# we will have a webiste which uses a gateway that then asks TF-serving. Then it will reply back to the gateway and this will send it back to the website.

# after, we will deploy everything to kubernetes (gateway+tf-serving).

# the gateway with FLASk will download the image, resize it and prepare for input. Then with the output post-process and send back
# the tf-serving will ust apply the model

#10.2 TensorFlow Serving

# first, download the model: 

wget https://github.com/DataTalksClub/machine-learning-zoomcamp/releases/download/chapter7-model/xception_v4_large_08_0.894.h5 -O clothing-model-v4.h5

import tensorflow as tf
from tensorflow import keras
model = keras.models.load_model('./clothing-model-v4.h5')
tf.saved_model.save(model, 'clothing_model')

# we can watch the clothing_model folder with tree

tree clothing_model

# we can also use to see what is inside recursively
ls -lhR clothing_model

# with this we can see what is inside the model
saved_model_cli show --dir clothing_model --all

# we are interested in a signature_def with inout and output. We need to know the name to invoke: serving_default
signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['input_8'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 299, 299, 3)
        name: serving_default_input_8:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['dense_7'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 10)
        name: StatefulPartitionedCall:0
  Method name is: tensorflow/serving/predict

# important
serving_default
input_8 - input
dense_7 - output

# now we will run docker
docker run -it --rm -p 8500:8500  -v "$(pwd)/clothing_model:/models/clothing-model/1" -e MODEL_NAME="clothing-model" tensorflow/serving:2.7.0

# now we will create a notebook.
# in the notbook, we will install gRPC whicih is a special protocol, works faster than json, and it is needed to communicate with tf-serving
 # then we write the code to communicate with a model deployed with tensorflow serving
 

 # 10.3 Creating a Pre-Processing Service
 
 # now convert this into python script
jupyter nbconvert --to script tf-serving-connect.ipynb
mv tf-serving-connect.py gateway.py
# and clean gatewa.py

# add more code for flask

# we create test.py
# and we run it. Remember that the docker image needs to be running and the gateway also.

# now we will put everything into pipenv

pipenv install grpcio==1.42.0 flask gunicorn keras-image-helper
# we do not install tensorflow because it is very heavy and we do not want to deploy it, that is why we use tf serving. But we have in the gateway tf imported, as in tf.make_tensor_proto. SO we will do it another way with tensorflow-cpu. Alexei has the code extracted from tf to only use make_tensor_proto>: https://github.com/alexeygrigorev/tensorflow-protobuf

pipenv install tensorflow-protobuf==2.7.0

#we also need to copy the code in the website and replace with the part where we use make_tensor_proto
# for that we create a file called proto.py and put the code there
# and in gateway.py we add 

from proto import np_to_protobuf

# and remove the function of np_to_protobuff from tf

# lets test it:
pipenv shell
#in the gateway add only the localhost part.
# it does not work, it says to downgrade protobuff package. I will do that

pipenv install protobuf==3.20.0


# 10.4 Docker-Compose

# now we want to run both gateway and tf serving together at the same time with docker.

# prepare docker image

#create image-model.dockerfile

docker build -t zoomcamp-10-model:xception-v4-001 -f image-model.dockerfile . 

# now we run it:
docker run -it --rm -p 8500:8500 zoomcamp-10-model:xception-v4-001

# now we can test it. in gateway uncomment again the localhost part and then run this
pipenv run python gateway.py

# it works, now we want to do create docker image also for the gateway. 
# uncomment the flask part in gateway.py

# create image-gateway.dockerfile
# build:
docker build -t zoomcamp-10-gateway:001 -f image-gateway.dockerfile .

# and run:
docker run -it --rm -p 9696:9696 zoomcamp-10-gateway:001

# so now we have both dockers running simoultaneosuly
# to test it:
python3 test.py
# it conplains saying that can not connect to all adresses
# that is because it can not reach the tensorflow serving. We have the test.py sending POT to localhost at 9696 but then flask is tring to send request to tfserving at 8500 but we have localhost in the gateway
# so we need to map it to the correct port in tf-serving container.
# to link, they need to live in the same network

# we do that with docker-Compose
# allows to run mmultiple docker containers and link together in a single network.
# install first.

# in my computer I am doing this:
sudo apt install gnome-terminal
sudo apt remove docker-desktop
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
download https://desktop.docker.com/linux/main/amd64/docker-desktop-4.25.2-amd64.deb?utm_source=docker&utm_medium=webreferral&utm_campaign=docs-driven-download-linux-amd64

sudo apt-get install ./docker-desktop-4.25.2-amd64.deb
#then I run this but dunnot it it is needed
systemctl --user start docker-desktop

# now we create a docer-compose.yaml, a file to put which dockers we want to work with.

# we also need to change the code for the localhost inside gateway. Instead localhpst we will point it to tfserving
# we will make it configurable. with environmental variables in os library.
# if we already have the env variable set then is fine and if not the code that we write will set it,.
# like this: 
host = os.getenv('TF_SERVING_HOST','localhost:8500')

#rebuild docker gateway
docker build -t zoomcamp-10-gateway:002 -f image-gateway.dockerfile .

#change the docker compose file adding 002 and also environment variables.

# now stop gateway and tf-serving
# run docker compose 
docker compose up # we need to also change version to string and no strings in the localhost part

docker ps #to check them
docker compose down #to stop

# docker compose it is very nice to test everything locally.
# now deploy to kubernetes


# 10.5 Introduction to Kubernetes

# kubernetes.io
# we can use kubernetes to deply docker containers. It deploys in the cloud and handles loads and other stuff making it easy for us.
# in the cluster we have NODEs and inside Nodes we have different PODs. Pods are docker containers. DEPLOYMENTS group the same PODs, the same docker image. PODs can have more cpu or more or less RAM. Then we have Gateway service and model service. Each service is reponsible for different functions.

# the gateway service is external and is the one seen. Called Load Balancer.
# the model service is internal. Cluster IP
# The first contact is done with the ingress, which then calls the gateway service. So this would be the entry point

# when we have many clients asking for many requests, Kubernetes starts more PODs, in each deployment. SO it can balance the load and scale up or down. The thing taken care of this is the HPA, Horizontal POD autoscaler

# 10.6 Deploying a Simple Service to Kubernetes

# first, we will create a simple Ping application in Flask.
# create a dir called ping, with a file called ping.py in it.
# int the same directory, install flask and gunicorn with pipenv
# but before get a empty Pipfile so it installes in the diretory (we have pipfile in the parent directory and then it would no install in this directory)
touch Pipfile
pipenv install flask gunicorn

#now create a DockerFile. And build it with a tag because local kubenerte class called Kind does not like latest tag
docker build -t ping:v001 .

# and run the image
docker run -it --rm -p 9696:9696 ping:v001

# test with
curl localhost:9696/ping

# it works. Now lets deploy it to kubernetes, but we need to install a couple of things.

# website: https://kubernetes.io/docs/tasks/tools/

# since we will later deploy kubernetes in AWS, we can also install from this website:
# https://docs.aws.amazon.com/eks/latest/userguide/install-kubectl.html

# download the binary to a folder we created in the root folder of the zoomcamp called bin

sudo chmod +x kubectl

# now, make kubectl callable from the $PATH

# write in .bashrc: export PATH="/home/ibai/work/MLzoomcamp/bin:${PATH}"

# now lets install Kind (to deploy kubernetes locally)
# https://kind.sigs.k8s.io/docs/user/quick-start/#installation

# download https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64 also in bin folder
wget https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
mv kind-linux-amd64 kind
chmod +x kind

#create a cluster with
kind create cluster

# now we need to configure de kubectl

kubectl cluster-info --context kind-kind
#@chekc that works with 
kubectl get service
kubectl get deployment
# if this gets no error means that we have kubernetes and kind installed an ready

# now lets create a deployment and a service
# with visual-studio we can install an extension to make this easier for kubernetes
# the extension is called kubernetes

# first we create the deployment
vim deployment.yaml # in visual studio it will create automatically if we write deployment

# metadata name = deployment name
# spec, containers, name = pod name
# inside the template, each pod has a label
# in the selector(deployment), all pods with app label the same, belong to this deployment
# replica adds more pods (1 + what we wrote in the replica) to the deployment








