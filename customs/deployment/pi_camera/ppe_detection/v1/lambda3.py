import os
import io
import json
import tarfile
import glob
import time
import logging

import base64

import boto3

s3 = boto3.client('s3')


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
    print("Writing file to s3")

    write_to_file("/tmp/photo.jpg", event['image'])
    output_file_path = event['company_name'] + "/" + event["device_id"] + "/" + str(event["time"].split()[0]) + "/" + str('_'.join(event["time"].split()[1].split(':'))) + ".jpg"

    with open("/tmp/photo.jpg", "rb") as f:
        s3.upload_fileobj(f, "pi-camera-ppe-detection-s3", output_file_path)

    print("file written to s3")

    return {
        "statusCode": 200
    }