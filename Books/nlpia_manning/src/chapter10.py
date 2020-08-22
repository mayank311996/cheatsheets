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
input_vocabulary = sorted(input_vocabulary)
output_vocabulary = sorted(output_vocabulary)

input_vocab_size = len(input_vocabulary)
output_vocab_size = len(output_vocabulary)

max_encoder_seq_length = max(
    [len(txt) for txt in input_texts]
)
max_decoder_seq_length = max(
    [len(txt) for txt in target_texts]
)

input_token_index = dict(
    [(char, i) for i, char in enumerate(input_vocabulary)]
)
target_token_index = dict(
    [(char, i) for i, char in enumerate(output_vocabulary)]
)

reverse_input_char_index = dict(
    [(i, char) for char, i in input_token_index.items()]
)
reverse_target_char_index = dict(
    [(i, char) for char, i in target_token_index.items()]
)

#########################################################################################
encoder_input_data = np.zeros(
    (
        len(input_texts),
        max_encoder_seq_length,
        input_vocab_size
    ),
    dtype="float32"
)

decoder_input_data = np.zeros(
    (
        len(input_texts),
        max_decoder_seq_length,
        output_vocab_size
    ),
    dtype="float32"
)

decoder_target_data = np.zeros(
    (
        len(input_texts),
        max_decoder_seq_length,
        output_vocab_size
    ),
    dtype="float32"
)

for i, (input_text, target_text) in enumerate(
    zip(input_texts, target_texts)
):
    for t, char in enumerate(input_text):
        encoder_input_data[
            i,
            t,
            input_token_index[char]
        ] = 1

    for t, char in enumerate(target_text):
        decoder_input_data[
            i,
            t,
            target_token_index[char]
        ] = 1
        if t > 0:
            decoder_target_data[
                i,
                t-1,
                target_token_index[char]
            ] = 1

#########################################################################################
















