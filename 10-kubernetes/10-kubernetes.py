

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



