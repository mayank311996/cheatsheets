import json
import boto3

s3 = boto3.client('s3')


def lambda_handler(event, context):
    print('------------------------')
    print(event)
    # 1. Iterate over each record
    try:
        for record in event['Records']:
            # 2. Handle event by type
            if record['eventName'] == 'INSERT':
                print('-----------------entering function 1')
                handle_insert(record)
        print('------------------------')
        return "Success!"
    except Exception as e:
        print(e)
        print('------------------------')
        return "Error"


def handle_insert(record):
    print('------------------------entering function 2')
    newImage = record['dynamodb']['NewImage']
    newDeviceId = newImage['device_id']['S']
    newTime = newImage['time']['S']
    newCompanyName = newImage['company']['S']
    newPredictions = newImage['predictions']['S']

    print('------------------------printing')
    print(newDeviceId)
    print(newTime)
    print(newCompanyName)
    print(newPredictions)
    print('------------------------end printing')

    event = json.loads(newPredictions)

    print("File loading from s3")
    input_file_path = newCompanyName + "/" + newDeviceId + "/" \
                      + str(newTime.split()[0]) + "/" + str(
        '_'.join(newTime.split()[1].split(':'))) + ".jpg"

    with open("/tmp/photo.jpg", 'wb') as f:
        s3.download_fileobj("pi-camera-ppe-detection-s3", input_file_path, f)
    print("File loaded from s3")

    print("Writing file to s3")
    output_file_path = newCompanyName + "/" + newDeviceId + "/" \
                      + str(newTime.split()[0]) + "/" + str(
        '_'.join(newTime.split()[1].split(':'))) + ".jpg"

    with open("/tmp/photo.jpg", "rb") as f:
        s3.upload_fileobj(f, "pi-camera-ppe-detection-prediction-s3", output_file_path)
    print("file written to s3")
