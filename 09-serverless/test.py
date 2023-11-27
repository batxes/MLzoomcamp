import requests

#url = "http://localhost:8080/2015-03-31/functions/function/invocations"
url = "https://y8dpr8i7l6.execute-api.eu-north-1.amazonaws.com/test/predict"

data = {'url': 'http://bit.ly/mlbookcamp-pants'} # I guess this could also be a path

result = requests.post(url, json=data).json()
print (result)
