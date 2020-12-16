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

company_name = st.text_input('Input company name (Example: fgc)')
device_id = st.text_input('Input device id (Example: fgc_raspberrypi)')
date = st.text_input('Input date (Format: YYYY/MM/DD, Example: 2020/12/14)')
hour = st.text_input('Input hour (Condition: UTC time, Example: 19)')

payload = json.dumps({
    "hour": hour,
    "date": date,
    "device_id": device_id,
    "company_name": company_name
})


# data_dict = {
#     "image": binary,
#     "time": dt_string
# }

response = requests.post("https://j91v1wvsx4.execute-api.us-east-2.amazonaws.com/dev", data=payload)
# print(response)
response_dict = response.json()

if response_dict["raw_list"]:
    for i, item in enumerate(response_dict["raw_list"]):
        base64_img_bytes = item.encode('utf-8')
        base64bytes = base64.b64decode(base64_img_bytes)
        st.image(base64bytes, caption='Raw' + str(i),
                 width=300)

if response_dict["prediction_list"]:
    for i, item in enumerate(response_dict["prediction_list"]):
        base64_img_bytes = item.encode('utf-8')
        base64bytes = base64.b64decode(base64_img_bytes)
        st.image(base64bytes, caption='Prediction' + str(i),
                 width=300)