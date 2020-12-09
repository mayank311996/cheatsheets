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

binary = open('test.jpg', 'rb').read()

# now = datetime.now()
# dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

# data_dict = {
#     "image": binary,
#     "time": dt_string
# }

response = requests.post("https://i9t9o58074.execute-api.us-east-2.amazonaws.com/dev", data=binary)
# print(response)
response_dict = response.json()
# print(data_dict)
