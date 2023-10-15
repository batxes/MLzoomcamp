import pickle
from flask import Flask
from flask import request #to get information in json
from flask import jsonify #to convert data into json


model_file = 'model1.bin'
dv_file = 'dv.bin'

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)
with open(dv_file, 'rb') as f_in:
    dv = pickle.load(f_in)

#customer = {"job": "retired", "duration": 445, "poutcome": "success"}

app = Flask('serving')

#add decorators. A way to add extra functionality
@app.route('/predict',methods=['POST']) # we use post because we are seding information to the server
def predict():
    customer = request.get_json()
    X = dv.transform([customer]) #convert the customer into feature matrix
    y_pred = model.predict_proba(X)[0, 1] #use the model to predict. We are interested in the second value, the probability
    result = {
            'churn_probability': float(y_pred), # this is also numpy variable, so make it python variable
            }
    return jsonify(result) #we also need to sent the information in JSON foramt

# when wwe send the info, we sent it in JSON format
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0',port=9696)


