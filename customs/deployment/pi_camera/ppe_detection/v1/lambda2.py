import os
import io
import json
import tarfile
import glob
import time
import logging

import base64

import boto3

reko = boto3.client('rekognition')


def write_to_file(save_path, base64str):
    """
    This function writes input image to temporary file for further processing
    :param save_path: Output path
    :param data: Input image
    :return: None
    """
    base64_img_bytes = base64str.encode('utf-8')
    base64bytes = base64.b64decode(base64_img_bytes)
    with open(save_path, "wb") as f:
        f.write(base64bytes)


def lambda_handler(event, context):
    """Lambda handler function
    Parameters
    ----------
    event: dict, required
        API Gateway Lambda Proxy Input Format
        Event doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html#api-gateway-simple-proxy-for-lambda-input-format
    context: object, required
        Lambda Context runtime methods and attributes
        Context doc: https://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html
    Returns
    ------
    API Gateway Lambda Proxy Output Format: dict
        Return doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html
    """
    print("Getting predictions from Rekognition")

    write_to_file("/tmp/photo.jpg", event['image'])
    with open("/tmp/photo.jpg", 'rb') as img_file:
        image = {'Bytes': img_file.read()}
    response_rekognition = reko.detect_protective_equipment(
        Image=image)

    print("Got predictions")

    print("Dumping response to DynamoDB")

    table = boto3.resource('dynamodb').Table(
        'pi_camera_ppe_detection_dynamodb')
    response_dynamo = table.put_item(
        Item={
            'device_id': event['device_id'],
            'company_name': event['company_name'],
            'time': event['time'],
            'predictions': json.dumps(response_rekognition)
        }
    )

    return {
        "statusCode": 200
    }