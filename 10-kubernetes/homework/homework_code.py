# code used for the homework:

docker pull svizor/zoomcamp-model:hw10

docker run -it --rm -p 9696:9696 svizor/zoomcamp-model:hw10

python3 q6_test.py

# {'get_credit': True, 'get_credit_probability': 0.726936946355423}

# Q1 Answer: 0.7269

kind --version
# Q2 Answer:  kind version 0.20.0

kind create cluster # gives error
#first I need to delete the previous cluster:
kind delete cluster

kubectl cluster-info

kubectl get service
NAME         TYPE        CLUSTER-IP   EXTERNAL-IP   PORT(S)   AGE
kubernetes   ClusterIP   10.96.0.1    <none>        443/TCP   24s

# Q3 Answer:  10.96.0.1

kind load docker-image svizor/zoomcamp-model:hw10

# Q4 Answer: kind load docker-image

# now I modified Image, Memory CPU and PORT with:
# Image: svizor/zoomcamp-model:hw10
# Memory: "128Mi"
# cpu: "100m"
# containerPort: 9696

# Q5 Answer: 9696

kubectl apply -f deployment.yaml

kubectl get pod
NAME                     READY   STATUS    RESTARTS   AGE
credit-86cdfbcb7-qprbj   1/1     Running   0          2s

# create service.yaml and replace:
# name: credit
# app: credit
# targetPort: 9696

# Q6 Answer: credit

kubectl apply -f service.yaml
