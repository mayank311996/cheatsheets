import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps
from tensorflow.keras.preprocessing.image import load_img
import time

##############################################################################
print(tf.__version__)
print(f"The following GPU devices are available: {tf.test.gpu_device_name()}")


##############################################################################
