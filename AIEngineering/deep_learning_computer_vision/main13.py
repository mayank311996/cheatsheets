########################################################################
# Data Collection for Image Classification using Python - Microsoft Bing Search
########################################################################

# pip install bing-image-downloader
# mkdir images
from bing_image_downloader import downloader
from IPython.display import Image
from six import BytesIO
import numpy as np
import glob
import os
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


#########################################################################################
downloader.download(
    "elephant in africa",
    limit=5,
    output_dir='images',
    adult_filter_off=True,
    force_replace=False
)
downloader.download(
    "kangaroo in australia",
    limit=5,
    output_dir='images',
    adult_filter_off=True,
    force_replace=False
)

# ls images/ -alrt
# ls 'images'/'elephant in africa'
# ls 'images'/'kangaroo in australia'

Image("images/elephant in africa/Image_5.jpg")  # from Ipython
Image("images/kangaroo in australia/Image_3.jpg")


#########################################################################################
def load_image_into_numpy_array(path):
    img_data = open(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width,
                                              3)).astype(np.uint8)


#########################################################################################
elephant_image_path = '/content/images/elephant in africa/*'
elephant_images_np = []
for iname in glob.glob(elephant_image_path):
    elephant_images_np.append(load_image_into_numpy_array(iname))

plt.rcParams['figure.figsize'] = [14, 7]
for idx, elephant_image_np in enumerate(elephant_images_np):
    plt.subplot(2, 3, idx + 1)
    plt.imshow(elephant_image_np)
plt.show()

#########################################################################################