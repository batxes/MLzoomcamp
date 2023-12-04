

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

# lets apply this to a kubernetes cluster
kubectl apply -f deployment.yaml
kubectl get deployment
kubectl get pod

# we can use decribe to see what happens
kubectl describe pod ping-deployment-86b45d75bf-7ml7l
# we got an error. Kind needs to register the image to kubernetes, for that:
kind load docker-image ping:v001

# and now we can see is running with:
kubectl get pod

# now test the deployment locally with port forwarding

kubectl port-forward ping-deployment-86b45d75bf-7ml7l 9696:9696
# not it is forwarding the port in the host machine to the port on the deployment
# now in another terminal
curl localhost:9696/ping
# this should show handling connection for 9696 from the deployment

#now lets create a service
vim service.yaml
# medatada name = name of the service
# selector app = which pods qualify for forwarding requests
# targetPort = the port the container shows
# port = service port, the one user using to communicate. 80

#now create the service:
kubectl apply -f service.yaml
kubectl get service

# the service type should be LoadBalancer
# the external IP of the kubernetes is fine, says None, but the Load balancer which is exposed should be changed. This one will be pending forever unless we modify. When using not locallly like kind but deployend, this will be changed. Now, we can do port forwarding and act like the IP is set.
# lets forward 8080 in our machine (80 will complain) and forward to 80 in the kubernetes
kubectl port-forward service/ping 8080:80
#ŧest it with:
curl localhost:8080/ping

# 10.7 Deploying TensorFlow models to kubernetes

# let's create first a folder, kube-config
mkdir kube-config
#inside, lets get a yaml file.
# the containerPort is 8500,for tensorflow-service listening requests
# before deploying, we need the image to be available in kind. DO that in root directory
kind load docker-image zoomcamp-10-model:xception-v4-001

# now apply yaml in the kube-config folder
kubectl apply -f model-deployment.yaml
kubectl get pod # not its running. If not working make CPU smaller, like 0.5
# if we need to remove a pod we can do:
kubectl delete -f ../ping/deployment.yaml

# to test clothing model, lets por forward again
kubectl port-forward tf-serving-clothing-model-7f555c49b5-6skpp 8500:8500

#now to test lets use gateway.py withut the flask
python3 gateway.py

#now that it works, we want to create a service for this deployment and apply
cd kube-config
kubectl apply -f model-service.yaml

kubectl get service
# now to test, again we can do port forwarding but now for the service
kubectl port-forward service/tf-serving-clothing-model 8500:8500
# and test:
python3 gateway.py

# at this time, the TF serving model is working, is ready. Now we need to deploy the gateway part
# first, load docker image
kind load docker-image zoomcamp-10-gateway:002

# now, in kube-config create the gateway-deployment.yaml
# we need to set the environment variable in this yaml file
env:
  - name: TF_SERVING_HOST
    value: tf-serving-clothing-model.default.svc.cluster.local:8500 

# the value is the name which follows a convention for kubernetes

# to check that it works, we log ing to one of the pods to execute some commands in
kubectl get pod
kubectl exec -it ping-deployment-86b45d75bf-7ml7l -- bash
# now we want to access the service from inside, using the url that we put in the environment value
# first we will use curl in the pod, but we need to install it first
apt update
apt install curl
# all of this in the POD, and then:
curl localhost:9696/ping
# now, lets go from the pod to the service and back to the pod
curl ping.default.svc.cluster.local:80/ping
# not, check the TF model
# but we can not use curl because the tf-serving model does not deal with http requests.
# to know if the model is listening we can use telnet
# lets install it also
telnet tf-serving-clothing-model.default.svc.cluster.local 8500

# it connects and there is something listening there for requests

# lets deply now, in the kube-config:
kubectl apply -f gateway-deployment.yaml

#check again
kubectl port-forward gateway-7d74767d4b-fk657 9696:9696
#and in another terminal check with 
python3 test.py

# now that we have the deployment, we want a service for this deployment
vim gateway-service.yaml
kubectl apply -f gateway-service.yaml #remember to add the type as load balance
kubectl get service

# the services are still pending for an IP: remenber in production, we would have an IP there and this IP would be the one that we would use in test.py or our python script that calls

# test again:
kubectl port-forward service/gateway 8080:80
python3 test.py # change the localhost port from 9696 to 8080

# when we deploy, we couuld have a problem that not all PODs get the same load, that is something that kubernetes with default settings does, it fails. For that google:
kubernetes load balancing grpc

#and there it says what to do (a website without TEARS, it says)

# 10.8 Deploying to EKS
# we will create a EKS cluster on AWS, publis the image to ECR and configure kubectl

# go to AWS konsole, EKS
# https://eu-north-1.console.aws.amazon.com/eks/home?region=eu-north-1#
# we can create a cluster there bu instead we will use eksctl
# https://eksctl.io/

# download to bin directory

# now, we can create a cluster with eksctl
# but rather we will create a cluster.yaml in kube-config folder. Check in https://eksctl.io/getting-started/
#then:
eksctl create cluster -f eks-config.yaml

# now we need to take our local images like gateway and tf-serving and publish them to ECR, the container service in AWS

#to do that:
aws ecr create-repository --repository-name mlzoomcamp-image
#here, we copy the repositryUri: 
302181200248.dkr.ecr.eu-north-1.amazonaws.com/mlzoomcamp-image

#meanwhile , execute this:
ACCOUNT_ID=302181200248
REGION=eu-north-1
REGISTRY_NAME=mlzoomcamp-image
PREFIX=${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REGISTRY_NAME}

GATEWAY_LOCAL=zoomcamp-10-gateway:002
GATEWAY_REMOTE=${PREFIX}:zoomcamp-10-gateway-002
docker tag ${GATEWAY_LOCAL} ${GATEWAY_REMOTE}

MODEL_LOCAL=zoomcamp-10-model:xception-v4-001
MODEL_REMOTE=${PREFIX}:zoomcamp-10-model-xception-v4-001
docker tag ${MODEL_LOCAL} ${MODEL_REMOTE}

$(aws ecr get-login --no-include-email)

docker push ${MODEL_REMOTE} 
docker push ${GATEWAY_REMOTE} 

#now we need to take the URI from above and put in the configuration of these docker images
#copy echo ${GATEWAY_REMOTE} to gateway-deployment.yaml

#replace
spec:
      containers:
      - name: gateway
        image: zoomcamp-10-gateway:002
#with:
        spec:
      containers:
      - name: gateway
        image: 302181200248.dkr.ecr.eu-north-1.amazonaws.com/mlzoomcamp-image:zoomcamp-10-gateway-002

# now the same with model-deployment.yaml

image: zoomcamp-10-model:xception-v4-001
# for this
image: 302181200248.dkr.ecr.eu-north-1.amazonaws.com/mlzoomcamp-image:zoomcamp-10-model-xception-v4-001

# now the EKS cluster is ready, we can check with:
kubectl get nodes

# now, lets apply all config files in kube-config to the cluster
# enter the kube-config dir and:
kubectl apply -f model-deployment.yaml
kubectl apply -f model-service.yaml
kubectl get pod # lets check the mode

NAME                                         READY   STATUS    RESTARTS   AGE
tf-serving-clothing-model-79c8fb579d-2vj5b   1/1     Running   0          37s

ibai@ibai-PC:~/work/MLzoomcamp/10-kubernetes/kube-config$ kubectl get service
NAME                        TYPE        CLUSTER-IP     EXTERNAL-IP   PORT(S)    AGE
kubernetes                  ClusterIP   10.100.0.1     <none>        443/TCP    13m
tf-serving-clothing-model   ClusterIP   10.100.97.18   <none>        8500/TCP   39s

#like always, we can check fist with port-forwarding:
kubectl port-forward service/tf-serving-clothing-model 8500:8500

# and the test works:
ibai@ibai-PC:~/work/MLzoomcamp/10-kubernetes$ python3 gateway.py

# now the gateways:
kubect apply -f gateway-deployment.yaml
kubect apply -f gateway-service.yaml
kubectl get pod

NAME                                         READY   STATUS    RESTARTS   AGE
gateway-5d658c7b4f-dzw6r                     1/1     Running   0          41s
tf-serving-clothing-model-79c8fb579d-2vj5b   1/1     Running   0          4m11s

ibai@ibai-PC:~/work/MLzoomcamp/10-kubernetes/kube-config$ kubectl get service
NAME                        TYPE           CLUSTER-IP      EXTERNAL-IP                                                                PORT(S)        AGE
gateway                     LoadBalancer   10.100.14.245   ab66f9963949549608e233f64dc81530-2144541766.eu-north-1.elb.amazonaws.com   80:32419/TCP   50s
kubernetes                  ClusterIP      10.100.0.1      <none>                                                                     443/TCP        17m
tf-serving-clothing-model   ClusterIP      10.100.97.18    <none>                                                                     8500/TCP       4m18s

# now you see we have an externalIP for the gateway
# if we execute this line we can see it is listening:
telnet ab66f9963949549608e233f64dc81530-2144541766.eu-north-1.elb.amazonaws.com 80

#test the gateway:
kubectl port-forward service/gateway 8080:80
python3 test.py

# it works!

# now we copy he URI and paste it in the URL variable of our test.py
url = "http://ab66f9963949549608e233f64dc81530-2144541766.eu-north-1.elb.amazonaws.com/predict"

Python3 test.pt #WORKS


# to stpo:
eksctl delete cluster --name mlzoomcamp-eks















