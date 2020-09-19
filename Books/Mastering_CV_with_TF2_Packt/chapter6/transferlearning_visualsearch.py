from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, \
    GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD, Adam

# For ResNet and VGG16
# from tensorflow.keras.applications.resnet50 import ResNet50, \
#    preprocess_input
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.applications.vgg16 import preprocess_input

##############################################################################
# ResNet and VGG16, img_width, img_height = 224, 224
img_width, img_height = 299, 299



