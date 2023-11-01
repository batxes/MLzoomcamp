import pickle
from flask import Flask
from flask import request #to get information in json
from flask import jsonify #to convert data into json

import xgboost as xgb


model_file = 'model_xgb_rain_prediction.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

print ("Model Loaded")

app = Flask('rain')

#add decorators. A way to add extra functionality

@app.route('/predict',methods=['POST']) # we use post because we are seding information to the server
def predict():
    day = request.get_json()
    X = dv.transform([day]) #convert day into feature matrix
    features = dv.get_feature_names_out()
    dtest = xgb.DMatrix(X, feature_names=features.tolist())
    y_pred = model.predict(dtest)[0] #use the model to predict. 
    rain_tomorrow = y_pred >= 0.5
    result = {
            'rain_probability': float(y_pred),
            'rain_tomorrow': bool(rain_tomorrow) # we need to turn into python boolean, because json does not work with numpy boolean
            }
    return jsonify(result) #we also need to sent the information in JSON foramt

# when wwe send the info, we sent it in JSON format

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0',port=9696)

