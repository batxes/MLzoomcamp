import requests                                                                     

#this is for the cloud
#host = "churn-serving-env.eba-pwhtrpxp.eu-north-1.elasticbeanstalk.com"
#url = f'http://{host}/predict'

#this is when we run in our machine
url = 'http://localhost:9696/predict'
#client = {"job": "unknown", "duration": 270, "poutcome": "failure"}
client = {"job": "retired", "duration": 445, "poutcome": "success"}
response = requests.post(url, json=client).json()

print (response)




