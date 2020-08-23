#########################################################################################
# Chapter 4
#########################################################################################

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers

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

















