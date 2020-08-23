#########################################################################################
# Chapter 4
#########################################################################################

import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers, \
    regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#########################################################################################
EPOCHS = 5
BATCH_SIZE = 128
VERBOSE = 1
OPTIMIZER = tf.keras.optimizers.Adam()
VALIDATION_SPLIT = 0.95

IMG_ROWS, IMG_COLS = 28, 28
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1)
NB_CLASSES = 10


def build(input_shape, classes):
    """
    Generates LeNet deep learning model
    :param input_shape: Input shape of images (rows, columns, channels)
    :param classes: Output classes
    :return: deep learning model
    """
    model = models.Sequential()
    model.add(
        layers.Convolution2D(
            20,
            (5, 5),
            activation='relu',
            input_shape=input_shape
        )
    )
    model.add(
        layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2)
        )
    )

    model.add(
        layers.Convolution2D(
            50,
            (5, 5),
            activation='relu'
        )
    )
    model.add(
        layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2)
        )
    )

    model.add(
        layers.Flatten()
    )
    model.add(
        layers.Dense(
            500,
            activation='relu'
        )
    )
    model.add(
        layers.Dense(
            classes,
            activation='softmax'
        )
    )

    return model


(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

X_train = X_train.reshape((60000, 28, 28, 1))
X_test = X_test.reshape((10000, 28, 28, 1))

X_train, X_test = X_train/255.0, X_test/255.0

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

y_train = tf.keras.utils.to_categorical(y_train, NB_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NB_CLASSES)

model = build(
    input_shape=INPUT_SHAPE,
    classes=NB_CLASSES
)

model.compile(
    loss='categorical_crossentropy',
    optimizer=OPTIMIZER,
    metrics=['accuracy']
)

model.summary()

callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs')
]

history = model.fit(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=VERBOSE,
    validation_split=VALIDATION_SPLIT,
    callbacks=callbacks
)

score = model.evaluate(
    X_test,
    y_test,
    verbose=VERBOSE
)

print("\nTest score:", score[0])
print("Test accuracy:", score[1])

#########################################################################################
IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32
BATCH_SIZE = 128
EPOCHS = 20
CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIM = tf.keras.optimizers.RMSprop()


def build(input_shape, classes):
    """
    Creates a deep learning model
    :param input_shape: Input image shape
    :param classes: Output classes
    :return: DL model
    """
    model = models.Sequential()
    model.add(
        layers.Convolution2D(
            32,
            (3, 3),
            activation='relu',
            input_shape=input_shape
        )
    )
    model.add(
        layers.MaxPooling2D(
            pool_size=(2, 2)
        )
    )
    model.add(
        layers.Dropout(0.25)
    )
    model.add(
        layers.Flatten()
    )
    model.add(
        layers.Dense(
            512,
            activation='relu'
        )
    )
    model.add(
        layers.Dropout(0.5)
    )
    model.add(
        layers.Dense(classes, activation='softmax')
    )

    return model


callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs')
]

model.compile(
    loss='categorical_crossentropy',
    optimizer=OPTIM,
    metrics=['accuracy']
)

model.fit(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT,
    verbose=VERBOSE,
    callbacks=callbacks
)

score = model.evaluate(
    X_test,
    y_test,
    batch_size=BATCH_SIZE,
    verbose=VERBOSE
)

print("\nTest score:", score[0])
print("Test accuracy:", score[1])


#########################################################################################
def build_model():
    """
    Generates a deep learning model with 3 convolutional
    and one dense block
    :return: DL model
    """
    model = models.Sequential()
    # 1st block
    model.add(
        layers.Convolution2D(
            32,
            (3, 3),
            padding='same',
            input_shape=x_train.shape[1:],
            activation='relu'
        )
    )
    model.add(
        layers.BatchNormalization()
    )
    model.add(
        layers.Convolution2D(
            32,
            (3, 3),
            padding='same',
            activation='relu'
        )
    )
    model.add(
        layers.BatchNormalization()
    )
    model.add(
        layers.MaxPooling2D(
            pool_size=(2, 2)
        )
    )
    model.add(
        layers.Dropout(0.2)
    )
    # 2nd block
    model.add(
        layers.Convolution2D(
            64,
            (3, 3),
            padding='same',
            activation='relu'
        )
    )
    model.add(
        layers.BatchNormalization()
    )
    model.add(
        layers.Convolution2D(
            64,
            (3, 3),
            padding='same',
            activation='relu'
        )
    )
    model.add(
        layers.BatchNormalization()
    )
    model.add(
        layers.MaxPooling2D(
            pool_size=(2, 2)
        )
    )
    model.add(
        layers.Dropout(0.3)
    )
    # 3rd block
    model.add(
        layers.Convolution2D(
            128,
            (3, 3),
            padding='same',
            activation='relu'
        )
    )
    model.add(
        layers.BatchNormalization()
    )
    model.add(
        layers.Convolution2D(
            128,
            (3, 3),
            padding='same',
            activation='relu'
        )
    )
    model.add(
        layers.BatchNormalization()
    )
    model.add(
        layers.MaxPooling2D(
            pool_size=(2, 2)
        )
    )
    model.add(
        layers.Dropout(0.4)
    )
    # dense
    model.add(
        layers.Flatten()
    )
    model.add(
        layers.Dense(
            CLASSES,
            activation='softmax'
        )
    )

    return model


model.summary()

EPOCHS = 50
NUM_CLASSES = 10
BATCH_SIZE = 64


def load_data():
    """
    Loads CIFAR10 dataset from tf.keras.datasets
    :return: CIFAR10 dataset
    """
    (x_train, y_train), (x_test, y_test) = \
    datasets.cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # normalize
    mean = np.mean(x_train, axis=(0, 1, 2, 3))
    std = np.std(x_train, axis=(0, 1, 2, 3))
    x_train = (x_train-mean)/(std+1e-7)
    x_test = (x_test-mean)/(std+1e-7)

    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

    return x_train, y_train, x_test, y_test


(x_train, y_train, x_test, y_test) = load_data()

model = build_model()
model.compile(
    loss='categorical_crossentropy',
    optimizer='RMSprop',
    metrics=['accuracy']
)

model.fit(
    x_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_test, y_test)
)
score = model.evaluate(
    x_test,
    y_test,
    batch_size=BATCH_SIZE
)
print("\nTest score:", score[0])
print("Test Accuracy", score[1])

#########################################################################################
# image augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
datagen.fit(x_train)

# train
model.fit_generator(
    datagen.flow(
        x_train,
        y_train,
        batch_size=BATCH_SIZE
    ),
    epochs=EPOCHS,
    verbose=VERBOSE,
    validation_data=(x_test, y_test)
)

# save to disk
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('model.h5')

# test
scores = model.evaluate(
    x_test,
    y_test,
    batch_size=128,
    verbose=1
)
print(f"\nTest result: {scores[0], scores[1]}")

#########################################################################################


























