#########################################################################################
# Chapter 9
#########################################################################################

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LSTM
from tensorflow.keras.models import model_from_json

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


















