import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from local_utils import detect_lp
from os.path import splitext, basename
from tensorflow.keras.models import model_from_json
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder
import glob
import os
from utils_plate import load_model
from utils_plate import get_plate
from utils_plate import emphasize_image
from utils_plate import sort_contours
from utils_plate import predict_from_model
import streamlit as st
from PIL import Image
import requests

# remove warning message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

##############################################################################
st.set_option('deprecation.showfileUploaderEncoding', False)
st.header("Upload here for License Plate Detection")

uploaded_file = st.file_uploader("Choose an image...",
                                 key="1")  # , type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image.save('test.jpg')
    st.image(image, caption='Uploaded Image for License Plate Detection',
             width=300)
    st.write("")
    st.write("Predicted:")

    # image = load_img(uploaded_file)
    # image = img_to_array(image)

    wpod_net_path = "wpod-net.json"
    wpod_net = load_model(wpod_net_path)

    # test_image_path = "1.jpg"
    vehicle, LpImg, cor = get_plate('test.jpg', wpod_net)

    plate_text = ""
    if len(LpImg):
        plate_image, binary, thre_mor = emphasize_image(LpImg)
        cv2.imwrite('binary.jpg', thre_mor)

        binary = open('binary.jpg', 'rb').read()
        response = requests.post("https://fpw5c4sbwe.execute-api.us-east-2.amazonaws.com/dev", data=binary)
        data_dict = response.json()
        print(data_dict)

        textDetections = data_dict["TextDetections"]
        for text in textDetections:
            if text['Confidence'] > 90.00:
                if text['Type'] == 'LINE':
                    plate_text += text['DetectedText']

    st.write('%s' % plate_text)

##############################################################################
