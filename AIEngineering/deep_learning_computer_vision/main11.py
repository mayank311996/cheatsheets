########################################################################
# Data Collection for Image Classification using Python - Google Images
########################################################################

# pip install git+https://github.com/Joeclinton1/google-images-download.git
from google_images_download import google_images_download
# The package from pip install google_images_download doesn't work with
# new google search. (remember in laptop it is installed through
# pip install google_images_download so change it later and download
# using github repo)
from six import BytesIO
from PIL import Image
import numpy as np
import glob
import os
import matplotlib
import matplotlib.pyplot as plt

#########################################################################################
response = google_images_download.googleimagesdownload()

arguments = {
    'keywords': 'polar bear in arctic, penguin in antarctica',
    'limit': 6,
    'format': 'jpg',
    'print_urls': True
}
paths = response.download(arguments)

# ls downloads/
# ls downloads/'polar bear in arctic'
# ls downloads/'penguin in antartica'


#########################################################################################
def load_image_into_numpy_array(path):
    img_data = open(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width,
                                              3)).astype(np.uint8)


#########################################################################################












