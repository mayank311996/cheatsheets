import json
from sklearn.externals import joblib

model_name = "model.joblib"
model = joblib.load(model_name)


def predict(event, context):
    body = {
        "message": "OK"
    }

    params = event["queryStringParameters"]

    medInc = float(params['medInc'])/100000
    houseAge = float(params['houseAge'])
    aveRooms = float(params['aveRooms'])
    aveBedrms = float(params['aveBedrms'])
    population = float(params['population'])
    aveOccup = float(params['aveOccup'])
    latitude = float(params['latitude'])
    longitude = float(params['longitude'])

    inputVector = [
        medInc,
        houseAge,
        aveRooms,
        aveBedrms,
        population,
        aveOccup,
        latitude,
        longitude
    ]

    data = [inputVector]
    predictPrice = model.predict(data)[0] * 100000
    predictPrice = round(predictPrice, 2)
    body['predictPrice'] = predictPrice

    response = {
        "statusCode": 200,
        "body": json.dumps(body),
        "headers": {
            "Access-Control-Allow-Origin": "*"
        }
    }

    return response


