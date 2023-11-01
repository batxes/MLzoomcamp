import requests

#this is when we run in our machine
url = "http://localhost:9696/predict"

#this is for the cloud
#host = "churn-serving-env.eba-pwhtrpxp.eu-north-1.elasticbeanstalk.com"
#url = f"http://{host}/predict"

day = {
 "Location":  "Albany",
 "MinTemp":  10,
 "MaxTemp":  15.5,
 "Rainfall":  5.0,
 "Evaporation":  2.1,
 "Sunshine":  12.4,
 "WindGustDir":  "W",
 "WindGustSpeed":  50.0,
 "WindDir9am":  "E",
 "WindDir3pm":  "E",
 "WindSpeed9am":  23.0,
 "WindSpeed3pm":  40.0,
 "Humidity9am":  50.0,
 "Humidity3pm":  23.0,
 "Pressure9am":  1020.7,
 "Pressure3pm":  1025.3,
 "Cloud9am":  3.0,
 "Cloud3pm": 6.0,
 "Temp9am":  10.0,
 "Temp3pm":  12.1,
 "RainToday":  1,
 "year":  2008,
 "month": "april",
 "day": 2}

response = requests.post(url, json=day).json()

print (response)

if response["rain_tomorrow"]:
    print("tomorrow will rain with a {} probability.".format(response["rain_probability"]))
else:
    print("tomorrow will NOT rain. Very low probability {}.".format(response["rain_probability"]))

