import requests

#this is when we run in our machine
url = 'http://localhost:9696/predict'

#this is for the cloud
host = "churn-serving-env.eba-pwhtrpxp.eu-north-1.elasticbeanstalk.com"
url = f'http://{host}/predict'

customer={
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "no",
    "phoneservice": "no",
    "multiplelines": "no_phone_service",
    "internetservice": "dsl",
    "onlinesecurity": "no",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "two_year",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "tenure": 10,
    "monthlycharges": 29.85,
    "totalcharges": 29.85
}

response = requests.post(url, json=customer).json()

print (response)

if response['churn']:
    print('sending email to', 'asdx-123d')
else:
    print('No need to send email to', 'asdx-123d')

