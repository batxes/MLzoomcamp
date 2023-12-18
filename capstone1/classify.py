from flask import Flask
from flask import request as flask_request #to get information in json
from flask import jsonify #to convert data into json

from tensorflow import keras
import numpy as np
import os
from io import BytesIO
from PIL import Image
from urllib import request

def download_image(url):
    print(url)
    with request.urlopen(url) as url:
        f = BytesIO(url.read())
    img = Image.open(f)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def prepare_input(x):
    return x / 255.0

model_file = 'final-map-model.h5'
model = keras.models.load_model(model_file)

print ("Model Loaded")

app = Flask('map')

#add decorators. A way to add extra functionality
@app.route('/classify', methods=['POST'])
def classify():
    url = flask_request.get_json()['url']
    print(url)

    img = download_image(url)
    img = prepare_image(img, target_size=(64, 64))

    x = np.array(img)
    X = np.array([x])

    X = prepare_input(X)

    pred = model.predict(X)

    class_pred_dict = {0: 'AnnualCrop',
         1: 'Forest',
         2: 'HerbaceousVegetation',
         3: 'Highway',
         4: 'Industrial',
         5: 'Pasture',
         6: 'PermanentCrop',
         7: 'Residential',
         8: 'River',
         9: 'SeaLake'}
    

    return jsonify(str(class_pred_dict[np.argmax(pred,axis=1)[0]]))


# when wwe send the info, we sent it in JSON format
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0',port=9696)
