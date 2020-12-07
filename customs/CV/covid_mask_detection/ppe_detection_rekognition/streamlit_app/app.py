import streamlit as st
import base64
import io
import requests, json
from PIL import Image, ImageDraw
import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
import random


def bytesioObj_to_base64encode(bytesObj):
    return base64.b64encode(bytesObj.read())

def base64encode_to_PILImage(base64encode):
    # base64_img_bytes = base64str.encode('utf-8')
    base64bytes = base64.b64decode(base64encode)
    bytesObj = io.BytesIO(base64bytes)
    img = Image.open(bytesObj)
    return img

st.set_option('deprecation.showfileUploaderEncoding', False)
st.markdown("<h1>COVID-19 Mask Detection App</h1><br>",
            unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image file")



if uploaded_file is not None:

    # image = Image.open(open(photo,'rb'))
    image = Image.open(uploaded_file)
    image.save('test.jpg')
    st.image(image, caption='Uploaded Image for Face Detection',
             width=300)
    st.write("")
    # In streamlit we will get a bytesIO object from the file_uploader
    # and we convert it to base64str for our FastAPI

    # base64encode = bytesioObj_to_base64encode(bytesObj)
    # print(base64encode)
    binary = open('test.jpg', 'rb').read()

    # We will also create the image in PIL Image format using this base64 str
    # Will use this image to show in matplotlib in streamlit
    # img = base64encode_to_PILImage(base64encode)

    response = requests.post("https://i9t9o58074.execute-api.us-east-2.amazonaws.com/dev", data=binary)
    # print(response)
    data_dict = response.json()
    # print(data_dict)

    fill_green = '#00d400'
    fill_red = '#ff0000'
    fill_yellow = '#ffff00'
    line_width = 3
    confidence = 80

    imgWidth, imgHeight = image.size
    draw = ImageDraw.Draw(image)

    for person in data_dict['Persons']:

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

                    # Check if confidence is lower than supplied value
                    # if ppe_item['CoversBodyPart']['Confidence'] < confidence:
                    #     # draw warning yellow bounding box within face mask bounding box
                    #     offset = line_width + line_width
                    #     points = (
                    #         (left + offset, top + offset),
                    #         (left + width - offset, top + offset),
                    #         ((left) + (width - offset),
                    #          (top - offset) + (height)),
                    #         (left + offset, (top) + (height - offset)),
                    #         (left + offset, top + offset)
                    #     )
                    #     draw.line(points, fill=fill_yellow, width=line_width)

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

    # image.show()
    st.image(image, caption='Mask detection',
             width=300)

    # st.markdown("<center><h1>App Result</h1></center>", unsafe_allow_html=True)
    # drawboundingbox(img, data_dict['boxes'], data_dict['classes'])
    # st.pyplot()
    # st.markdown("<center><h1>FastAPI Response</h1></center><br>",
    #             unsafe_allow_html=True)
    # st.write(data_dict)