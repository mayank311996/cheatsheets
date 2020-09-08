########################################################################
# Hands on TensorFlow Computer Vision with TPU - Part 1
########################################################################

import tensorflow as tf
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt

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
# mkdir /tmp/logs

resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
    'grpc://' + os.environ['COLAB_TPU_ADDR']
)  # this fetches the TPU address
tf.contrib.distribute.initialize_tpu_system(resolver)  # initialize TPU
strategy = tf.contrib.distribute.TPUStrategy(resolver)

with strategy.scope():
    model = create_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )

    logdir = os.path.join(
        "/tmp/logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        logdir,
        histogram_freq=1,
        profile_batch=3
    )

    history = model.fit(
        x_train.astype(np.float32),
        y_train.astype(np.float32),
        epochs=20,
        steps_per_epoch=50,
        validation_data=(
            x_test.astype(np.float32),
            y_test.astype(np.float32)
        ),
        validation_freq=5,
        callbacks=[tensorboard_callback]
    )

plt.figure(figsize=(6, 8))
plt.plot(history.history['sparse_categorical_accuracy'])
plt.plot(history.history['val_sparse_categorical_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_acc', 'val_acc'])
plt.show()

# %tensorboard --logdir /tmp/logs

#########################################################################################
















