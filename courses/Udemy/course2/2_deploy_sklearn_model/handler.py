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


# this is just to check function locally
def do_main():
    event = {
        "queryStringParameters": {
            "medInc": 200000,
            "houseAge": 10,
            "aveRooms": 4,
            "aceBedrms": 1,
            "population": 800,
            "aveOccup": 3,
            "latitude": 32.54,
            "longitude": -121.72
        }
    }

    response = predict(event, None)
    body = json.loads(response['body'])
    print('Price', body["predictedPrice"])

    with open('event.json', 'w') as event_file:
        event_file.write(json.dumps(event))


do_main()























