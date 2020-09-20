import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, \
    GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import pathlib

##############################################################################
img_width, img_height = 299, 299
AUTOTUNE = tf.data.experimental.AUTOTUNE
NUM_EPOCHS = 5
batchsize = 10
num_train_images = 1350
num_val_images = 156

base_model = InceptionV3(
    weigths='imagenet',
    include_top=False,
    input_shape=(img_height, img_width, 3)
)

train_dir = '../train'
val_dir = '../validation'
train_dir = pathlib.Path(train_dir)
val_dir = pathlib.Path(val_dir)
output_dir = '../output'

train_ds = tf.data.Dataset.list_files(str(train_dir/'*/*'))
val_ds = tf.data.Dataset.list_files(str(val_dir/'*/*'))
class_list = ["bed", "chair", "sofa"]

for f in train_ds.take(5):
    print(f.numpy())


##############################################################################


















