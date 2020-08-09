#########################################################################################
# Chapter 7
#########################################################################################

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Dense, Dropout, Activation

#########################################################################################
model = Sequential()
model.add(
    Conv1D(
        filters=16,
        kernel_size=3,
        padding='same',
        activation='relu',
        strides=1,
        input_shape=(100, 300)
    )
)
