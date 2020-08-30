import json
import os
import io
import boto3
from urllib.request import Request, urlopen

runtime = boto3.client('runtime.sagemaker')
SAGEMAKER_ENDPOINT_NAME = os.environ['SAGEMAKER_ENDPOINT_NAME']


def detect_diabetes(event, context):
    body = {
        "message": "ok",
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

    body = ','.join([str(item) for item in input_data])

    response = runtime.invoke_endpoint(
        EndpointName='v1-xgboost-diabetes',
        ContentType='text/csv',
        Body=body.encode('utf-8')
    )

    result = response['Body'].read().decode('utf-8')

    response = {
        'statusCode': 200,
        'body': json.dumps(round(float(result))),
        'headers': {
            "Access-Control-Allow-Origin": "*"
        }
    }

    return response
