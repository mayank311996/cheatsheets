#########################################################################################
# Chapter 10
#########################################################################################

import sys
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LSTM, \
    Embedding, Activation, GRU, Input
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import RMSprop

#########################################################################################
encoder_inputs = Input(
    shape=(None, input_vocab_size)
)
encoder = LSTM(
    num_neurons,
    return_state=True
)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = (state_h, state_c)

#########################################################################################
