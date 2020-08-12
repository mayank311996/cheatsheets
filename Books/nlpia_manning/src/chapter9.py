#########################################################################################
# Chapter 9
#########################################################################################

import sys
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LSTM, Embedding, Activation
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import RMSprop
from nltk.corpus import gutenberg

#########################################################################################
MAXLEN = 400
BATCH_SIZE = 32
EMBEDDING_DIMS = 300
EPOCHS = 2
NUM_NEURONS = 50

model = Sequential()
model.add(
    LSTM(
        NUM_NEURONS,
        return_sequences=True,
        input_shape=(MAXLEN, EMBEDDING_DIMS)
    )
)
model.add(
    Dropout(0.2)
)
model.add(
    Flatten()
)
model.add(
    Dense(1, activation="sigmoid")
)
model.compile(
    "rmsprop",
    "binary_crossentropy",
    metrics=["accuracy"]
)
print(model.summary())


#########################################################################################
def pre_process_data(filepath):
    """
    Function to pre-process the data. Assigns label and shuffles
    the data.
    :param filepath: File location
    :return: shuffled dataset with labels
    """
    positive_path = os.path.join(filepath, 'pos')
    negative_path = os.path.join(filepath, 'neg')
    pos_label = 1
    neg_label = 0
    dataset = []

    for filename in glob.glob(os.path.join(positive_path, '*.txt')):
        with open(filename, 'r') as f:
            dataset.append((pos_label, f.read()))
    for filename in glob.glob(os.path.join(negative_path, '*.txt')):
        with open(filename, 'r') as f:
            dataset.append((neg_label, f.read()))
    shuffle(dataset)

    return dataset


def tokenize_and_vectorize(dataset):
    """
    Function to tokenize and vectorize the given dataset
    :param dataset: Input data with second column as text
    :return: Vectorized dataset
    """
    tokenizer = TreebankWordTokenizer()
    vectorized_data = []
    for sample in dataset:
        tokens = tokenizer.tokenize(sample[1])
        sample_vecs = []
        for token in tokens:
            try:
                sample_vecs.append(word_vectors[token])
            except KeyError:
                pass  # No matching token in the Google w2v vocab
        vectorized_data.append(sample_vecs)
    return vectorized_data


def collect_expected(dataset):
    """
    This function takes care of sentiment column
    :param dataset: Input data with first column as sentiment
    :return: Sentiments in order
    """
    expected = []
    for sample in dataset:
        expected.append(sample[0])

    return expected


def pad_trunc(data, maxlen):
    """
    This takes of padding and truncation of input list of tokens
    :param data: input data
    :param maxlen: Max length for padded list
    :return: Padded and truncated list of tokens
    """
    new_data = []
    # Create a vector of Os the length of our word vectors
    zero_vector = []
    for _ in range(len(data[0][0])):
        zero_vector.append(0.0)

    for sample in data:
        if len(sample) > maxlen:
            temp = sample[:maxlen]
        elif len(sample) < maxlen:
            temp = sample
            # Append the appropriate number of 0 vectors to the list
            additional_elems = maxlen - len(sample)
            for _ in range(additional_elems):
                temp.append(zero_vector)
        else:
            temp = sample
        new_data.append(temp)
    return new_data


#########################################################################################
dataset = pre_process_data('<path to your downloaded file>/aclimdb/train')

vectorized_data = tokenize_and_vectorize(dataset)
expected = collect_expected(dataset)

split_point = int(len(vectorized_data)*0.8)

x_train = vectorized_data[:split_point]
y_train = expected[:split_point]
x_test = vectorized_data[split_point:]
y_test = expected[split_point:]

maxlen = 400
batch_size = 32
embedding_dims = 300
epochs = 2

x_train = pad_trunc(x_train, maxlen)
x_test = pad_trunc(x_test, maxlen)

x_train = np.reshape(x_train, (len(x_train), maxlen, embedding_dims))
y_train = np.array(y_train)
x_test = np.reshape(x_test, (len(x_test), maxlen, embedding_dims))
y_test = np.array(y_test)

MAXLEN = 400
BATCH_SIZE = 32
EMBEDDING_DIMS = 300
EPOCHS = 2
NUM_NEURONS = 50

model = Sequential()
model.add(
    LSTM(
        NUM_NEURONS,
        return_sequences=True,
        input_shape=(MAXLEN, EMBEDDING_DIMS)
    )
)
model.add(
    Dropout(0.2)
)
model.add(
    Flatten()
)
model.add(
    Dense(1, activation="sigmoid")
)
model.compile(
    "rmsprop",
    "binary_crossentropy",
    metrics=["accuracy"]
)
print(model.summary())

#########################################################################################
model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    validation_data=(x_test, y_test)
)

model_structure = model.to_json()
with open("lstm_model1.json", "w") as json_file:
    json_file.write(model_structure)
model.save_weights("lstm_weights1.h5")

#########################################################################################
with open("lstm_model1.json", "r") as json_file:
    json_string = json_file.read()
model = model_from_json(json_string)
model.load_weights("lstm_weights1.h5")

sample_1 = "I hate that the dismal weather had me down for so long, " \
           "when will it break! Ugh, when does happiness return? " \
           "The sun is blinding and the puffy clouds are too thin. " \
           "I can't wait for the weekend."

vec_list = tokenize_and_vectorize([(1, sample_1)])
test_vec_list = pad_trunc(vec_list, maxlen)
test_vec = np.reshape(test_vec_list, (
    len(test_vec_list),
    maxlen,
    embedding_dims
))
model.predict(test_vec)
model.predict_classes(test_vec)


#########################################################################################
def test_len(data, maxlen):
    """
    Function to check suitability of maxlen
    :param data: input data
    :param maxlen: max len of list of tokens
    :return: none
    """
    total_len = truncated = exact = padded = 0
    for sample in data:
        total_len += len(sample)
        if len(sample) > maxlen:
            truncated += 1
        elif len(sample) < maxlen:
            padded += 1
        else:
            exact += 1

    print(f"Padded: {padded}")
    print(f"Truncated: {truncated}")
    print(f"Equal: {exact}")
    print(f"Avg length: {total_len/len(data)}")


dataset = pre_process_data("./aclimdb/train")
vectorized_data = tokenize_and_vectorize(dataset)
test_len(vectorized_data, 400)

#########################################################################################
dataset = pre_process_data('<path to your downloaded file>/aclimdb/train')

vectorized_data = tokenize_and_vectorize(dataset)
expected = collect_expected(dataset)

split_point = int(len(vectorized_data)*0.8)

x_train = vectorized_data[:split_point]
y_train = expected[:split_point]
x_test = vectorized_data[split_point:]
y_test = expected[split_point:]

maxlen = 200
batch_size = 32
embedding_dims = 300
epochs = 2

x_train = pad_trunc(x_train, maxlen)
x_test = pad_trunc(x_test, maxlen)

x_train = np.reshape(x_train, (len(x_train), maxlen, embedding_dims))
y_train = np.array(y_train)
x_test = np.reshape(x_test, (len(x_test), maxlen, embedding_dims))
y_test = np.array(y_test)

MAXLEN = 200
BATCH_SIZE = 32
EMBEDDING_DIMS = 300
EPOCHS = 2
NUM_NEURONS = 50

model = Sequential()
model.add(
    LSTM(
        NUM_NEURONS,
        return_sequences=True,
        input_shape=(MAXLEN, EMBEDDING_DIMS)
    )
)
model.add(
    Dropout(0.2)
)
model.add(
    Flatten()
)
model.add(
    Dense(1, activation="sigmoid")
)
model.compile(
    "rmsprop",
    "binary_crossentropy",
    metrics=["accuracy"]
)
print(model.summary())

model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    validation_data=(x_test, y_test)
)

model_structure = model.to_json()
with open("lstm_model7.json", "w") as json_file:
    json_file.write(model_structure)
model.save_weights("lstm_weights7.h5")

#########################################################################################
# dataset = pre_process_data("./aclimdb/train")
# expected = collect_expected(dataset)


def avg_len(data):
    """
    Calculates avg num of characters present in a dataset per sentence.
    :param data: input dataset
    :return: avg num of characters
    """
    total_len = 0
    for sample in data:
        total_len += len(sample[1])
    return total_len/len(data)


print(avg_len(dataset))


#########################################################################################
def clean_data(data):
    """
    This function converts data sentences into lower case,
    replace unknowns with UNK, and listify
    :param data: input dataset
    :return: listified data
    """
    new_data = []
    VALID = 'abcdefghijklmnopqrstuvwxyz0123456789"\'?!.,:; '
    for sample in data:
        new_sample = []
        for char in sample[1].lower():
            if char in VALID:
                new_sample.append(char)
            else:
                new_sample.append('UNK')
        new_data.append(new_sample)
    return new_data


# listified_data = clean_data(dataset)


#########################################################################################
def char_pad_trunc(data, maxlen=1500):
    """
    Truncate data to maxlen or add padding
    :param data: input dataset
    :param maxlen: max len for num of characters in a sentence
    :return: Truncated and padded list of characters
    """
    new_datset = []
    for sample in data:
        if len(sample) > maxlen:
            new_data = sample[:maxlen]
        elif len(sample) < maxlen:
            pads = maxlen - len(sample)
            new_data = sample + ["PAD"]*pads
        else:
            new_data = sample
        new_datset.append(new_data)
    return new_datset


#########################################################################################
def create_dicts(data):
    """
    Creates char to indices and indices to char dictionaries
    for mapping
    :param data: input dataset
    :return: mapping dictionaries
    """
    chars = set()
    for sample in data:
        chars.update(set(sample))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    return char_indices, indices_char


#########################################################################################
def onehot_encode(dataset, char_indices, maxlen=1500):
    """
    Function to one-hot-encode the tokens
    :param dataset: input dataset (list of lists of tokens)
    :param char_indices: dictionary of char to indices mapping
    {key = character, value = index to use encoding vector}
    :param maxlen: max len of for num of characters in a sentence
    :return: Encoded numpy array of shape (samples, tokens,
    encoding length)
    """
    X = np.zeros((len(dataset), maxlen, len(char_indices.keys())))
    for i, sentence in enumerate(dataset):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1

    return X


#########################################################################################
dataset = pre_process_data("./aclimdb/train")
expected = collect_expected(dataset)
listified_data = clean_data(dataset)

common_length_data = char_pad_trunc(listified_data, maxlen=1500)
char_indices, indices_char = create_dicts(common_length_data)
encoded_data = onehot_encode(common_length_data, char_indices, 1500)

split_point = int(len(encoded_data)*0.8)

x_train = encoded_data[:split_point]
y_train = expected[:split_point]
x_test = encoded_data[split_point:]
y_test = expected[split_point:]

#########################################################################################
NUM_NEURONS = 40
MAXLEN = 1500

model = Sequential()
model.add(
    LSTM(
        NUM_NEURONS,
        return_sequences=True,
        input_shape=(MAXLEN, len(char_indices.keys()))
    )
)
model.add(
    Dropout(0.2)
)
model.add(
    Flatten()
)
model.add(
    Dense(1, activation="sigmoid")
)

model.compile(
    "rmsprop",
    "binary_crossentropy",
    metrics=["accuracy"]
)
model.summary()

#########################################################################################
BATCH_SIZE = 12
EPOCHS = 10

model.fit(
    x_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_test, y_test)
)

model_structure = model.to_json()
with open("char_lstm_model3.json", "w") as json_file:
    json_file.write(model_structure)
model.save_weights("char_lstm_weights3.h5")

#########################################################################################
print(gutenberg.fileids())

text = ""
for txt in gutenberg.fileids():
    if "shakespeare" in txt:
        text += gutenberg.raw(txt).lower()

chars = sorted(list(set(text)))

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

print(f"corpus length: {len(text)}, total chars: {len(chars)}")
print(text[:500])

#########################################################################################
maxlen = 40
step = 3
sentences = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i:i + maxlen])
    next_chars.append(text[i + maxlen])

print("nb sequences: ", len(sentences))

X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

#########################################################################################
model = Sequential()

model.add(
    LSTM(
        128,
        input_shape=(maxlen, len(chars))
    )
)
model.add(
    Dense(
        len(chars)
    )
)
model.add(
    Activation(
        "softmax"
    )
)

optimizer = RMSprop(
    lr=0.01
)

model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizer
)
model.summary()

epochs = 6
batch_size = 128
model_structure = model.to_json()
with open("shakes_lstm_model.json", "w") as json_file:
    json_file.write(model_structure)
for i in range(5):
    model.fit(
        X,
        y,
        batch_size=batch_size,
        epochs=epochs
    )
    model.save_weights(f"shakes_lstm_weights_{i+1}.h5")


#########################################################################################
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)/temperature
    exp_preds = np.exp(preds)
    preds = exp_preds/np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


#########################################################################################
start_index = random.randint(0, len(text) - maxlen - 1)
for diversity in [0.2, 0.5, 1.0]:
    print()
    print("----- diversity:", diversity)
    generated = ""
    sentence = text[start_index: start_index + maxlen]
    generated += sentence
    print("----- Generating with seed: " + sentence)
    sys.stdout.write(generated)

    for i in range(400):
        x = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.0
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]
        generated += next_char
        sentence = sentence[1:] + next_char
        sys.stdout.write(next_char)
        sys.stdout.flush()
    print()

#########################################################################################













