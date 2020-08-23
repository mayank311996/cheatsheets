#########################################################################################
# Chapter 1
#########################################################################################

import tensorflow as tf
import numpy as np
from tf.keras.regularizers import l2, activity_l2

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





















