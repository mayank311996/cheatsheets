#########################################################################################
# Chapter 10
#########################################################################################

import sys
import random
import numpy as np
from nlpia.loaders import get_data
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
decoder_inputs = Input(
    shape=(None, output_vocab_size)
)
decoder_lstm = LSTM(
    num_neurons,
    return_sequences=True,
    return_state=True
)
decoder_outputs, _, _ = decoder_lstm(
    decoder_inputs,
    initial_state=encoder_states
)
decoder_dense = Dense(
    output_vocab_size,
    activation="softmax"
)
decoder_outputs = decoder_dense(decoder_outputs)

#########################################################################################
model = Model(
    inputs=[encoder_inputs, decoder_inputs],
    outputs=decoder_outputs
)

#########################################################################################
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy'
)
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=batch_size,
    epochs=epochs
)

#########################################################################################
encoder_model = Model(
    inputs=encoder_inputs,
    outputs=encoder_states
)

#########################################################################################
thought_input = [
    Input(shape=(num_neurons,)),
    Input(shape=(num_neurons,))
]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs,
    initial_state=thought_input
)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    inputs=[decoder_inputs] + thought_input,
    output=[decoder_outputs] + decoder_states
)

#########################################################################################
thought = encoder_model.predict(
    input_seq
)

while not stop_condition:
    output_tokens, h, c = decoder_model.predict(
        [target_seq] + thought
    )

#########################################################################################
df = get_data('moviedialog')
input_texts, target_texts = [], []
input_vocabulary = set()
output_vocabulary = set()
start_token = '\t'
stop_token = '\n'
max_training_samples = min(25000, len(df) - 1)

for input_text, target_text in zip(df.statement, df.reply):
    target_text = start_token + target_text + stop_token

    input_texts.append(input_text)
    target_texts.append(target_text)

    for char in input_text:
        if char not in input_vocabulary:
            input_vocabulary.add(char)
    for char in target_text:
        if char not in output_vocabulary:
            output_vocabulary.add(char)

#########################################################################################
























