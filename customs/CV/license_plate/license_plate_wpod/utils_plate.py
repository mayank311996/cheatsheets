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


def get_plate(image_path, Dmax=608, Dmin = 608):
    """
    Function to extract license plate from the given picture
    :param image_path: Path to input image
    :param Dmax: Max boundary dimension
    :param Dmin: Min boundary dimension
    :return: Original image, extracted image, coordinates of the plate
    """
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return vehicle, LpImg, cor


def emphasize_image(extracted_image):
    """
    Function to reduce noise and emphasize features of license plate
    :param extracted_image: Extracted image by get_plate function
    :return: Processed emphasized images binary and thre_mor
    """
    # Scales, calculates absolute values, and converts the result to 8-bit.
    plate_image = cv2.convertScaleAbs(extracted_image[0], alpha=(255.0))

    # convert to grayscale and blur the image
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Applied inversed thresh_binary
    binary = cv2.threshold(blur, 180, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)

    return binary, thre_mor


def sort_contours(cnts, reverse=False):
    """
    Function to grab the contour of each digit from left to right
    :param cnts: Contours
    :param reverse: Reverse (True or False)
    :return: sorted contours
    """
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse)
                                )
    return cnts

