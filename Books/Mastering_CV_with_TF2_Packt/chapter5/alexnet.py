import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.datasets import cifar10

##############################################################################
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)

##############################################################################
model = Sequential()

# Conv and Pool 1
model.add(
    Conv2D(
        96,
        (11, 11),
        input_shape=x_train.shape[1:],
        padding="same",
        kernel_regularizer=l2(l2_reg)
    )
)
model.add(
    BatchNormalization()
)
model.add(
    Activation(
        'relu'
    )
)
model.add(
    MaxPooling2D(
        pool_size=(2, 2)
    )
)

# Conv and Pool 2
model.add(
    Conv2D(
        256,
        (5, 5),
        # input_shape=x_train.shape[1:],
        padding="same",
        # kernel_regularizer=l2(l2_reg)
    )
)
model.add(
    BatchNormalization()
)
model.add(
    Activation(
        'relu'
    )
)
model.add(
    MaxPooling2D(
        pool_size=(2, 2)
    )
)

# Conv and Pool 3
model.add(
    ZeroPadding2D(
        (1, 1)
    )
)
model.add(
    Conv2D(
        512,
        (3, 3),
        # input_shape=x_train.shape[1:],
        padding="same",
        # kernel_regularizer=l2(l2_reg)
    )
)
model.add(
    BatchNormalization()
)
model.add(
    Activation(
        'relu'
    )
)
model.add(
    MaxPooling2D(
        pool_size=(2, 2)
    )
)

# Conv and Pool 4
model.add(
    ZeroPadding2D(
        (1, 1)
    )
)
model.add(
    Conv2D(
        1024,
        (3, 3),
        # input_shape=x_train.shape[1:],
        padding="same",
        # kernel_regularizer=l2(l2_reg)
    )
)
model.add(
    BatchNormalization()
)
model.add(
    Activation(
        'relu'
    )
)

# Conv and Pool 5
model.add(
    ZeroPadding2D(
        (1, 1)
    )
)
model.add(
    Conv2D(
        1024,
        (3, 3),
        # input_shape=x_train.shape[1:],
        padding="same",
        # kernel_regularizer=l2(l2_reg)
    )
)
model.add(
    BatchNormalization()
)
model.add(
    Activation(
        'relu'
    )
)
model.add(
    MaxPooling2D(
        pool_size=(2, 2)
    )
)

# Fully Connected 1
model.add(
    Flatten()
)
model.add(
    Dense(
        3072
    )
)
model.add(
    BatchNormalization()
)
model.add(
    Activation(
        'relu'
    )
)
model.add(
    Dropout(
        0.5
    )
)

# Fully Connected 2
model.add(
    Dense(
        4096
    )
)
model.add(
    BatchNormalization()
)
model.add(
    Activation(
        'relu'
    )
)
model.add(
    Dropout(
        0.5
    )
)

# Fully Connected 3
model.add(
    Dense(
        2
    )
)
model.add(
    BatchNormalization()
)
model.add(
    Activation(
        'softmax'
    )
)

##############################################################################
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adadelta(),
    metrics=['accuracy']
)

##############################################################################








