import json
import pickle

model = pickle.load(open("./Random_Forest.pkl", "rb"))


def predict(event, context):
    body = {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "input": event
    }

    params = event['queryStringParameters']

    pregnancies = float(params['pregnancies'])
    glucose = float(params['glucose'])
    bp = float(params['bp'])
    skinthickness = float(params['skinthickness'])
    insulin = float(params['insulin'])
    bmi = float(params['bmi'])
    diabetespedigreefunction = float(params['diabetespedigreefunction'])
    age = float(params['age'])

    input_data = [[
        pregnancies,
        glucose,
        bp,
        skinthickness,
        insulin,
        bmi,
        diabetespedigreefunction,
        age
    ]]

    pred = model.predict(input_data)[0]

    body['prediction'] = pred

    response = {
        "statusCode": 200,
        "body": json.dumps(body),
        "headers": {
            "Access-Control-Allow-Origin": "*"
        }
    }

    return response

# the access-control-allow-origin is used inorder to account for
# cross origin references this happens when your fronend and backend are
# different domains or something
    # Use this code if you don't use the http event with the LAMBDA-PROXY
    # integration
