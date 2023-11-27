# coding: utf-8

import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor
from io import BytesIO
from urllib import request

from PIL import Image

preprocessor = create_preprocessor('xception', target_size=(150,150))

#interpreter = tflite.Interpreter(model_path='bees-model.tflite')
interpreter = tflite.Interpreter(model_path='bees-wasps-v2.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index'] #index says which part of the model is the input
output_index = interpreter.get_output_details()[0]['index'] # and this the output

# url = "https://habrastorage.org/webt/rt/d9/dh/rtd9dhsmhwrdezeldzoqgijdg8a.jpeg"

def predict(url):
    X = preprocessor.from_url(url)
    interpreter.set_tensor(input_index, X)
    interpreter.invoke() # with this, we moved our images X through all the layers of the model, 
                        #  and we have the result in the output
    preds = interpreter.get_tensor(output_index)

    float_predictions = preds[0].tolist()

    #return dict(zip(classes,preds[0]))
    return float_predictions

# this code is necessary for lambda
def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result
