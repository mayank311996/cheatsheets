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

# remove warning message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

##############################################################################
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

    if len(LpImg):
        plate_image, binary, thre_mor = emphasize_image(LpImg)
        cv2.imwrite('binary.jpg', binary)

    cont, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    print(cont)

    # creat a copy version "test_roi" of plat_image to draw bounding box
    test_roi = plate_image.copy()

    # Initialize a list which will be used to append charater image
    crop_characters = []

    # define standard width and height of character
    digit_w, digit_h = 30, 60

    for c in sort_contours(cont):
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h / w
        if 1 <= ratio <= 3.5:  # Only select contour with defined ratio
            if h / plate_image.shape[0] >= 0.5:
                # Select contour which has the height larger
                # than 50% of the plate
                # Draw bounding box arroung digit number
                cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Sperate number and gibe prediction
                curr_num = thre_mor[y:y + h, x:x + w]
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                _, curr_num = cv2.threshold(curr_num, 220, 255,
                                            cv2.THRESH_BINARY +
                                            cv2.THRESH_OTSU)
                crop_characters.append(curr_num)

    for i in range(len(crop_characters)):
        cv2.imwrite('trial' + str(i) + '.jpg', crop_characters[i])

    # Load model architecture, weight and labels
    json_file = open('MobileNets_character_recognition.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("License_character_recognition_weight.h5")
    print("[INFO] Model loaded successfully...")

    labels = LabelEncoder()
    labels.classes_ = np.load('license_character_classes.npy')
    print("[INFO] Labels loaded successfully...")

    final_string = ''
    for i, character in enumerate(crop_characters):
        title = np.array2string(predict_from_model(character, model, labels))
        final_string += title.strip("'[]")

    st.write('%s' % final_string)

##############################################################################
