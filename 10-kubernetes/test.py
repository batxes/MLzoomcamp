import requests

#url = "http://localhost:9696/predict"
url = "http://localhost:8080/predict"
url = "http://ab66f9963949549608e233f64dc81530-2144541766.eu-north-1.elb.amazonaws.com/predict"

data = {'url': 'http://bit.ly/mlbookcamp-pants'} # I guess this could also be a path

result = requests.post(url, json=data).json()
print (result)
