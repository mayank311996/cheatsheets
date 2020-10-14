import fastbook
from fastbook import *
from fastai.vision.widgets import *
import cv2
import streamlit as st

from dataset import torch, os, LocalDataset, transforms, np, get_class, \
    num_classes, preprocessing, Image, m, s
from config import *

# remove warning message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

##############################################################################
st.header("Upload here for Car Make and Model Detection")

uploaded_file = st.file_uploader("Choose an image...",
                                 key="1")  # , type="jpg")

learn_inf = load_learner(path/'export.pkl')

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image.save('test.jpg')
    st.image(image, caption='Uploaded Image for License Plate Detection',
             width=300)
    st.write("")
    st.write("Predicted:")

    # image = load_img(uploaded_file)
    # image = img_to_array(image)

    img = cv2.imread('test.jpg')
    # img = img_to_array(image)
    pred, pred_idx, probs = learn_inf.predict(img)

    st.write(f"Prediction: {pred}")
    st.write(f"Probability: {probs[pred_idx]}")

##############################################################################