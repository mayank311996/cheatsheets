#########################################################################################
# Chapter 2
#########################################################################################

import numpy as np
import pandas as pd

#########################################################################################
sentence = "Thomas Jefferson began building Monticello at the age of 26."
sentence.split()
str.split(sentence)


#########################################################################################
token_sequence = str.split(sentence)
vocab = sorted(set(token_sequence))
', '.join(vocab)
num_tokens = len(token_sequence)
vocab_size = len(vocab)
onehot_vectors = np.zeros((num_tokens,
                           vocab_size), int)
for i, word in enumerate(token_sequence):
    onehot_vectors[i, vocab.index(word)] = 1
' '.join(vocab)
print(onehot_vectors)

pd.DataFrame(onehot_vectors, columns=vocab)
# df[df == 0] = ''


#########################################################################################
sentence_bow = {}
for token in sentence.split():
    sentence_bow[token] = 1
sorted(sentence_bow.items())


#########################################################################################
















