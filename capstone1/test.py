# coding: utf-8

import requests

#this is when we run in our machine
url = "http://localhost:9696/classify"
url = "http://0.0.0.0:9696/classify"

#this is for the cloud
#host = "midterm-project-env.eba-myupfmwp.eu-north-1.elasticbeanstalk.com"
#url = f"http://{host}/predict"

data = {
        #"url": "https://github.com/batxes/MLzoomcamp/blob/main/capstone1/test/Industrial/Industrial_1.jpg"
  "url": "https://i.imgur.com/6n71Nae.jpg"
}

response = requests.post(url, json=data).json()
print(response)

