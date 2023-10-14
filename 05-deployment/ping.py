from flask import Flask

app = Flask('ping')

#add decorators. A way to add extra functionality

@app.route('/ping',methods=['GET'])
def ping():
    return "PONG"

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0',port=9696)
