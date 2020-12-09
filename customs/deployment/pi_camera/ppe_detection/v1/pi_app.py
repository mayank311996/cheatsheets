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
from datetime import datetime

##############################################################################


def bytesioObj_to_base64str(bytesObj):
    return base64.b64encode(bytesObj).decode("utf-8")


def base64str_to_Image(base64str):
    base64_img_bytes = base64str.encode('utf-8')
    base64bytes = base64.b64decode(base64_img_bytes)
    with open('trial.jpg', "wb") as f:
      f.write(base64bytes)


now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

binary = open('test.jpg', 'rb').read()

base64str = bytesioObj_to_base64str(binary)

payload = json.dumps({
    "base64str": base64str,
    "time": dt_string,
    "device_id": "fgc_raspberrypi",
    "company_name": "fgc"
})


# data_dict = {
#     "image": binary,
#     "time": dt_string
# }

response = requests.post("https://zc4mbbbjzc.execute-api.us-east-2.amazonaws.com/dev", data=payload)
# print(response)
response_dict = response.json()
print(response_dict)
