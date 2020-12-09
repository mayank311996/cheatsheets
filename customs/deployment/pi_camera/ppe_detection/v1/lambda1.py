import os
import io
import json
import tarfile
import glob
import time
import logging

import base64

import boto3

lambda_client = boto3.client('lambda')


def lambda_handler(event, context):
    print("Sending stuff to Lambda_2")
    event = json.loads(event['body'])
    inputParams = {
        'device_id': event['device_id'],
        'company_name': event['company_name'],
        'time': event['time'],
        'image': event['base64str']
    }

    response = lambda_client.invoke(
        FunctionName='arn:aws:lambda:us-east-2:882207224999:function:pi_camera_ppe_detection_v1_L_2',
        InvocationType='RequestResponse',
        Payload=json.dumps(inputParams)
    )

    return {
        "statusCode": 200,
        "body": json.dumps({'Status': "Successful"})
    }