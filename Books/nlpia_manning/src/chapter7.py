#########################################################################################
# Chapter 7
#########################################################################################

import glob
import os
from random import shuffle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Dense, Dropout, Activation
from nltk.tokenize import TreebankWordTokenizer
from gensim.models.keyedvectors import KeyedVectors
from nlpia.loaders import get_data

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


#########################################################################################
dataset = pre_process_data('<path to your downloaded file>/aclimdb/train')
print(dataset[0])

#########################################################################################
word_vectors = get_data("w2v", limit=200000)


def tokenize_and_vectorize(dataset):
    """
    Function to tokenize and vectorize the given dataset
    :param dataset: Input data with second column as text
    :return: Vectorized dataset
    """
    tokenizer = TreebankWordTokenizer()
    vectorized_data = []
    expected = []
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


vectorized_data = tokenize_and_vectorize(dataset)
expected = collect_expected(dataset)

#########################################################################################
split_point = int(len(vectorized_data)*0.8)

x_train = vectorized_data[:split_point]
y_train = expected[:split_point]
x_test = vectorized_data[split_point:]
y_test = expected[split_point:]

#########################################################################################
maxlen = 400
batch_size = 32
embedding_dims = 300
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 2


#########################################################################################
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
x_train = pad_trunc(x_train, maxlen)
x_test = pad_trunc(x_test, maxlen)

x_train = np.reshape(x_train, (len(x_train), maxlen, embedding_dims))
y_train = np.array(y_train)
x_test = np.reshape(x_test, (len(x_test), maxlen, embedding_dims))
y_test = np.array(y_test)

#########################################################################################











