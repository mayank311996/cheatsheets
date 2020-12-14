import json


def lambda_handler(event, context):
    print('------------------------')
    print(event)
    # 1. Iterate over each record
    try:
        for record in event['Records']:
            # 2. Handle event by type
            if record['eventName'] == 'INSERT':
                handle_insert(record)
        print('------------------------')
        return "Success!"
    except Exception as e:
        print(e)
        print('------------------------')
        return "Error"

def handle_insert(record):

    newImage = record['dynamodb']['NewImage']
    newDeviceId = newImage['device_id']['S']
    newTime = newImage['time']['S']
    newCompanyName = newImage['company']['S']
    newPredictions = newImage['predictions']['S']

    print(newDeviceId)
    print(newTime)
    print(newCompanyName)
    print(newPredictions)