import json
import boto3
import base64

s3 = boto3.resource('s3')


def lambda_handler(event, context):

    print("Sending stuff to Lambda_2")
    # event = json.loads(event['body'])

    device_id = event['device_id']
    company_name = event['company_name']
    date = event['date']
    hour = event['hour']

    find_prefix = company_name + "/" + device_id + "/" \
                      + date + "/" + hour + "_"

    bucket_raw = s3.Bucket("pi-camera-ppe-detection-s3")
    bucket_predictions = s3.Bucket("pi-camera-ppe-detection-prediction-s3")

    raw_list = []
    prediction_list = []

    for obj in bucket_raw.objects.filter(Prefix=find_prefix):
        print(obj)
        if obj.size:
            binary = obj.get()['Body'].read()
            base64str = bytesioObj_to_base64str(binary)
            raw_list.append(base64str)

    for obj in bucket_predictions.objects.filter(Prefix=find_prefix):
        if obj.size:
            binary = open("s3://pi-camera-ppe-detection-prediction-s3/" + obj.key, 'rb').read()
            base64str = bytesioObj_to_base64str(binary)
            prediction_list.append(base64str)

    return {
        "statusCode": 200,
        "body": json.dumps({'raw_list': raw_list,
                            'prediction_list': prediction_list})
    }


def bytesioObj_to_base64str(bytesObj):
    return base64.b64encode(bytesObj).decode("utf-8")