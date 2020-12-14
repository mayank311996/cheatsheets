import json
import boto3
from PIL import Image, ImageDraw

s3 = boto3.client('s3')


def lambda_handler(event, context):
    print('------------------------')
    print(event)
    # 1. Iterate over each record

    for record in event['Records']:
        # 2. Handle event by type
        if record['eventName'] == 'INSERT':
            print('-----------------entering function 1')
            handle_insert(record)
        else:
            print("something is wrong as record cannot be modified")
    print('------------------------')
    return {
        "statusCode": 200
    }


def handle_insert(record):
    print('------------------------entering function 2')
    newImage = record['dynamodb']['NewImage']
    newDeviceId = newImage['device_id']['S']
    newTime = newImage['time']['S']
    newCompanyName = newImage['company_name']['S']
    newPredictions = newImage['predictions']['S']

    print('------------------------printing')
    print(newDeviceId)
    print(newTime)
    print(newCompanyName)
    print(newPredictions)
    print('------------------------end printing')

    newPredictions_dict = json.loads(newPredictions)

    print("File loading from s3")
    input_file_path = newCompanyName + "/" + newDeviceId + "/" \
                      + str(newTime.split()[0]) + "/" + str(
        '_'.join(newTime.split()[1].split(':'))) + ".jpg"

    with open("/tmp/photo.jpg", 'wb') as f:
        s3.download_fileobj("pi-camera-ppe-detection-s3", input_file_path, f)
    print("File loaded from s3")

    print("Processing image based in rekognition response")
    image = Image.open(open("/tmp/photo.jpg", 'rb'))
    fill_green = '#00d400'
    fill_red = '#ff0000'
    fill_yellow = '#ffff00'
    line_width = 3
    confidence = 80

    imgWidth, imgHeight = image.size
    draw = ImageDraw.Draw(image)

    for person in newPredictions_dict['Persons']:

        found_mask = False

        for body_part in person['BodyParts']:
            ppe_items = body_part['EquipmentDetections']

            for ppe_item in ppe_items:
                # found a mask
                if ppe_item['Type'] == 'FACE_COVER':
                    fill_color = fill_green
                    found_mask = True
                    # check if mask covers face
                    if ppe_item['CoversBodyPart']['Value'] == False:
                        fill_color = fill = '#ff0000'
                    # draw bounding box around mask
                    box = ppe_item['BoundingBox']
                    left = imgWidth * box['Left']
                    top = imgHeight * box['Top']
                    width = imgWidth * box['Width']
                    height = imgHeight * box['Height']
                    points = (
                        (left, top),
                        (left + width, top),
                        (left + width, top + height),
                        (left, top + height),
                        (left, top)
                    )
                    draw.line(points, fill=fill_color, width=line_width)

        if found_mask == False:
            # no face mask found so draw red bounding box around body
            box = person['BoundingBox']
            left = imgWidth * box['Left']
            top = imgHeight * box['Top']
            width = imgWidth * box['Width']
            height = imgHeight * box['Height']
            points = (
                (left, top),
                (left + width, top),
                (left + width, top + height),
                (left, top + height),
                (left, top)
            )
            draw.line(points, fill=fill_red, width=line_width)
    image.save("/tmp/processed.jpg")
    print("Done processing image")

    print("Writing file to s3")
    output_file_path = newCompanyName + "/" + newDeviceId + "/" \
                       + str(newTime.split()[0]) + "/" + str(
        '_'.join(newTime.split()[1].split(':'))) + ".jpg"

    with open("/tmp/processed.jpg", "rb") as f:
        s3.upload_fileobj(f, "pi-camera-ppe-detection-prediction-s3",
                          output_file_path)
    print("file written to s3")
