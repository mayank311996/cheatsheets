import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from local_utils import detect_lp
from os.path import splitext, basename
from keras.models import model_from_json
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder
import glob
import os

# remove warning message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


##############################################################################
def load_model(path):
    """
    This function loads pre-trained model from json and h5 files
    :param path: Path to any one of the file (json or h5)
    :return: Loaded model
    """
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)


def preprocess_image(image_path, resize=False):
    """
    Function to pre-process and resize image to be fed into the model for
    plate extraction
    :param image_path: Path to input image
    :param resize: Resize (True or False)
    :return: pre-processed image
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224, 224))
    return img
