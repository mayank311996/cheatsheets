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
from tensorflow.keras.models import Model

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
BATCH_SIZE = 64
EPOCHS = 100
NUM_NEURONS = 256

encoder_inputs = Input(
    shape=(None, input_vocab_size)
)
encoder = LSTM(
    NUM_NEURONS,
    return_state=True
)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(
    shape=(None, output_vocab_size)
)
decoder_lstm = LSTM(
    NUM_NEURONS,
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

model = Model(
    [encoder_inputs, decoder_inputs],
    decoder_outputs
)

model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['acc']
)

model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1
)

#########################################################################################
encoder_model = Model(encoder_inputs, encoder_states)
thought_input = [
    Input(shape=(NUM_NEURONS,)),
    Input(shape=(NUM_NEURONS,))
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
def decode_sequence(input_seq):
    """
    Function to generate the response (decoded sequence)
    :param input_seq: Input sequence (one-hot-encoded)
    :return: returns generated sequence by decoder
    """
    thought = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1, output_vocab_size))
    target_seq[0, 0, target_token_index[stop_token]] = 1
    stop_condition = False
    generated_sequence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + thought
        )

        generated_token_idx = np.argmax(output_tokens[0, -1, :])
        generated_char = reverse_target_char_index[generated_token_idx]
        generated_sequence += generated_char

        if (generated_char == stop_token or
        len(generated_sequence) > max_decoder_seq_length):
            stop_condition = True

        target_seq = np.zeros((1, 1, output_vocab_size))
        target_seq[0, 0, generated_token_idx] = 1
        thought = [h, c]

    return generated_sequence


#########################################################################################





























