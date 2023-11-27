# coding: utf-8

import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor

preprocessor = create_preprocessor('xception', target_size=(299,299))

interpreter = tflite.Interpreter(model_path='clothing-model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index'] #index says which part of the model is the input
output_index = interpreter.get_output_details()[0]['index'] # and this the output

classes = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants',
    'shirt',
    'shoes',
    'shorts',
    'skirt',
    't-shirt'
]

# url = 'http://bit.ly/mlbookcamp-pants' 

def predict(url):
    X = preprocessor.from_url(url)
    interpreter.set_tensor(input_index, X)
    interpreter.invoke() # with this, we moved our images X through all the layers of the model, 
                        #  and we have the result in the output
    preds = interpreter.get_tensor(output_index)

    float_predictions = preds[0].tolist()

    #return dict(zip(classes,preds[0]))
    return dict(zip(classes,float_predictions))

# this code is necessary for lambda
def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result








