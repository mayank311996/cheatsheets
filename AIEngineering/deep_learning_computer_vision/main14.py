########################################################################
# Hands on TensorFlow Computer Vision with TPU - Part 1
########################################################################

import tensorflow as tf
import numpy as np
import os
import datetime

#########################################################################################
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist. \
    load_data()

tf.config.experimental.list_physical_devices()
print(x_train.shape)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print(x_train.shape)


#########################################################################################
def create_model():
    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.Convolution2D(
            32,
            5,
            padding='same',
            activation='relu',
            input_shape=(28, 28, 1)
        )
    )
    model.add(
        tf.keras.layers.MaxPooling2D()
    )
    model.add(
        tf.keras.layers.Dropout(0.25)
    )

    model.add(
        tf.keras.layers.Convolution2D(
            64,
            5,
            padding='same',
            activation='relu'
        )
    )
    model.add(
        tf.keras.layers.MaxPooling2D()
    )
    model.add(
        tf.keras.layers.Dropout(0.25)
    )

    model.add(
        tf.keras.layers.Convolution2D(
            128,
            5,
            padding='same',
            activation='relu'
        )
    )
    model.add(
        tf.keras.layers.MaxPooling2D()
    )
    model.add(
        tf.keras.layers.Dropout(0.25)
    )

    model.add(
        tf.keras.layers.Convolution2D(
            256,
            5,
            padding='same',
            activation='relu'
        )
    )
    model.add(
        tf.keras.layers.MaxPooling2D()
    )
    model.add(
        tf.keras.layers.Dropout(0.25)
    )

    model.add(
        tf.keras.layers.Flatten()
    )
    model.add(
        tf.keras.layers.Dense(512)
    )
    model.add(
        tf.keras.layers.Activation('relu')
    )
    model.add(
        tf.keras.layers.Dropout(0.25)
    )
    model.add(
        tf.keras.layers.Dense(10)
    )
    model.add(
        tf.keras.layers.Activation('softmax')
    )

    return model


#########################################################################################










