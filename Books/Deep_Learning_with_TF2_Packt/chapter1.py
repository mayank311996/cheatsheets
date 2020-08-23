#########################################################################################
# Chapter 1
#########################################################################################

import tensorflow as tf
import numpy as np
from tf.keras.regularizers import l2, activity_l2
from tensorflow.keras import datasets, layers, models, preprocessing
import tensorflow_datasets as tfds

#########################################################################################
EPOCHS = 200
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2
RESHAPED = 784

# loading and splitting data
mnist = tf.keras.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# reshaping data
X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(60000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalizing input
X_train /= 255.0
X_test /= 255.0

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# One-hot-encoding
Y_train = tf.keras.utils.to_categorical(Y_train, NB_CLASSES)
Y_test = tf.keras.utils.to_categorical(Y_test, NB_CLASSES)

# build the model
model = tf.keras.models.Sequential()
model.add(
    keras.layers.Dense(
        NB_CLASSES,
        input_shape=(RESHAPED,),
        name='dense_layer',
        activation='softmax'
    )
)

# compiling the model
model.compile(
    optimizer='SGD',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# training the model
model.fit(
    X_train,
    Y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=VERBOSE,
    validation_split=VALIDATION_SPLIT
)

# evaluate the model
test_loss, test_acc = model.evaluate(X_test, Y_test)
print("Test accuracy:", test_acc)

#########################################################################################
EPOCHS = 50
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2
RESHAPED = 784

# loading and splitting data
mnist = tf.keras.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# reshaping data
X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(60000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalizing input
X_train /= 255.0
X_test /= 255.0

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# One-hot-encoding
Y_train = tf.keras.utils.to_categorical(Y_train, NB_CLASSES)
Y_test = tf.keras.utils.to_categorical(Y_test, NB_CLASSES)

# build the model
model = tf.keras.models.Sequential()
model.add(
    keras.layers.Dense(
        N_HIDDEN,
        input_shape=(RESHAPED,),
        name='dense_layer',
        activation='relu'
    )
)
model.add(
    keras.layers.Dense(
        N_HIDDEN,
        name='dense_layer2',
        activation='relu'
    )
)
model.add(
    keras.layers.Dense(
        NB_CLASSES,
        name='dense_layer3',
        activation='softmax'
    )
)

model.summary()

# compiling the model
model.compile(
    optimizer='SGD',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# training the model
model.fit(
    X_train,
    Y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=VERBOSE,
    validation_split=VALIDATION_SPLIT
)

# evaluate the model
test_loss, test_acc = model.evaluate(X_test, Y_test)
print("Test accuracy:", test_acc)

#########################################################################################
EPOCHS = 200
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2
RESHAPED = 784
DROPOUT = 0.3

# loading and splitting data
mnist = tf.keras.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# reshaping data
X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(60000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalizing input
X_train /= 255.0
X_test /= 255.0

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# One-hot-encoding
Y_train = tf.keras.utils.to_categorical(Y_train, NB_CLASSES)
Y_test = tf.keras.utils.to_categorical(Y_test, NB_CLASSES)

# build the model
model = tf.keras.models.Sequential()
model.add(
    keras.layers.Dense(
        N_HIDDEN,
        input_shape=(RESHAPED,),
        name='dense_layer',
        activation='relu'
    )
)
model.add(keras.layers.Dropout(DROPOUT))
model.add(
    keras.layers.Dense(
        N_HIDDEN,
        name='dense_layer2',
        activation='relu'
    )
)
model.add(keras.layers.Dropout(DROPOUT))
model.add(
    keras.layers.Dense(
        NB_CLASSES,
        name='dense_layer3',
        activation='softmax'
    )
)

model.summary()

# compiling the model
model.compile(
    optimizer='SGD',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# training the model
model.fit(
    X_train,
    Y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=VERBOSE,
    validation_split=VALIDATION_SPLIT
)

# evaluate the model
test_loss, test_acc = model.evaluate(X_test, Y_test)
print("Test accuracy:", test_acc)

#########################################################################################
model.compile(
    optimizer='RMSProp',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#########################################################################################
model.add(
    Dense(
        64,
        input_dim=64,
        W_regularizer=l2(0.01),
        activity_regularizer=activity_l2(0.01)
    )
)

#########################################################################################
MAX_LEN = 200
N_WORDS = 10000
DIM_EMBEDDING = 256
EPOCHS = 20
BATCH_SIZE = 500


def load_data():
    """
    This function pads the input sequences to uniform dimension
    :return: Returns padded data with labels
    """
    # load data
    (X_train, y_train), (X_test, y_test) = \
    datasets.imdb.load_data(num_words=N_WORDS)
    # pad sequences
    X_train = preprocessing.sequence.pad_sequences(
        X_train,
        maxlen=MAX_LEN
    )
    X_test = preprocessing.sequence.pad_sequences(
        X_test,
        maxlen=MAX_LEN
    )

    return (X_train, y_train), (X_test, y_test)


def build_model():
    """
    Builds DL model for sentiment analysis using one
    embedding layer
    :return: Deep Learning model
    """
    model = models.Sequential()
    model.add(
        layers.Embedding(
            N_WORDS,
            DIM_EMBEDDING,
            input_length=MAX_LEN
        )
    )
    model.add(
        layers.Dropout(0.3)
    )
    model.add(
        layers.GlobalMaxPooling1D()
    )
    model.add(
        layers.Dense(
            128,
            activation='relu'
        )
    )
    model.add(
        layers.Dropout(0.5)
    )
    model.add(
        layers.Dense(
            1,
            activation='sigmoid'
        )
    )

    return model


(X_train, y_train), (X_test, y_test) = load_data()
model = build_model()
model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

score = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test)  # wrong way maybe
)

score = model.evaluate(
    X_test,
    y_test,
    batch_size=BATCH_SIZE
)
print("\nTest score:", score[0])
print("Test accuracy:", score[1])
# predictions = model.predict(X)

#########################################################################################























