import requests

url = "http://localhost:9696/predict"

data = {'url': 'http://bit.ly/mlbookcamp-pants'} # I guess this could also be a path

result = requests.post(url, json=data).json()
print (result)
