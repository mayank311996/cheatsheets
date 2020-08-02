########################################################################
# End to End Image Classification using TensorFlow 2.0 (In Colab using TPUs)
########################################################################
import tensorflow as tf
import numpy as np
import os
import datetime
import tensorflow_datasets as tfds

# to use tpu's data should be on local colab machine or google cloud storage
datasets, info = tfds.load(
    name='fashion_mnist',
    with_info=True,
    as_supervised=True,
    split=["train", "test"],
    batch_size=-1  # loads data into local machine RAM
)

img_train, info_train = tfds.load(name="fashion_mnist", with_info=True, split="test")
tfds.show_examples(info_train, img_train)

tf.config.experimental.list_physical_devices()

train, test = tfds.as_numpy(datasets[0]), tfds.as_numpy(datasets[1])

x_train, y_train = train[0]/255, tf.one_hot(train[1], 10)
x_test, y_test = test[0]/255, tf.one_hot(test[1], 10)

x_test, x_val = x_test[5000:], x_test[:5000]
y_test, y_val = y_test[5000:], y_test[:5000]

# x_train.shape
print(x_test.shape)
print(x_val.shape)


def create_model():
    """
    Creates a deep CNN model
    :return: CNN model
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, 5, padding="same", activation="relu", input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Conv2D(64, 5, padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Conv2D(128, 5, padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Conv2D(256, 5, padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10))
    model.add(tf.keras.layers.Activation("softmax"))
    return model



















