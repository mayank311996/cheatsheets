########################################################################
# Object Detection - Data Collection and Bounding Boxes using Python
########################################################################


# pip install git+https://github.com/Joeclinton1/google-images-download.git
from google_images_download import google_images_download
# The package from pip install google_images_download doesn't work with
# new google search. (remember in laptop it is installed through
# pip install google_images_download so change it later and download
# using github repo)
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import glob
import os
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, Javascript
from IPython.display import Image as IPyImage
from object_detection.utils import visualization_utils as viz_utils
from onject_detection.utils import colab_utils
import tensorflow as tf
# %matplotlib inline

#########################################################################################
response = google_images_download.googleimagesdownload()

arguments = {
    'keywords': 'alone polar bear in artic',
    'limit': 6,
    'format': 'jpg',
    'print_urls': True
}
paths = response.download(arguments)

# ls downloads/'alone polar bear in artic'
# git clone --depth 1 https://github.com/tensorflow/models

# %%bash
# cd models/research/
# protoc object_detection/protos/*.proto --python_out=.
# cp object_detection/packages/tf2/setup.py
# python -m pip install .


#########################################################################################
def load_image_into_numpy_array(path):
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width,
                                              3)).astype(np.uint8)


#########################################################################################












