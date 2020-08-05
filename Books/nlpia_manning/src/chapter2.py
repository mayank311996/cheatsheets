#########################################################################################
# Chapter 2
#########################################################################################

import numpy as np
import pandas as pd
import re
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import TreebankWordTokenizer
from nltk.util import ngrams
import nltk
nltk.download("stopwords")

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
df = pd.DataFrame(pd.Series(dict([(token, 1) for token in sentence.split()
                                  ])), columns=["sent"]).T

sentences = "Thomas Jefferson began building Monticello at the" \
            "age of 26.\n"
sentences += "Construction was done mostly by local masons and" \
             "carpenters.\n"
sentences += "He moved into the South Pavilion in 1770.\n"
sentences += "Turning Monticello into a neoclassical masterpiece" \
             "was Jefferson's obsession."
corpus = {}
for i, sent in enumerate(sentences.split('\n')):
    corpus[f'sent{i}'] = dict((tok, 1) for tok in sent.split())
df = pd.DataFrame.from_records(corpus).fillna(0).astype(int).T


#########################################################################################
df = df.T
df.sent0.dot(df.sent1)
df.sent0.dot(df.sent2)
df.sent0.dot(df.sent3)

print([(k, v) for (k, v) in (df.sent0 & df.sent3).items() if v])


#########################################################################################
sentence = "Thomas Jefferson began building Monticello at the" \
           "age of 26."
tokens = re.split(r'[-\s.,;!?]+', sentence)

pattern = re.compile(r'[-\s.,;!?]+')
tokens = pattern.split(sentence)
print([x for x in tokens if x and x not in '-\t\n.,;!?'])


#########################################################################################
tokenizer = RegexpTokenizer(r'\w+|$[0-9.]+|\S+')
tokenizer.tokenize(sentence)


#########################################################################################
sentence = "Monticello wan't designated as UNESCO World Heritage" \
           "Site until 1987."
tokenizer = TreebankWordTokenizer()
tokenizer.tokenize(sentence)


#########################################################################################
pattern = re.compile(r'[-\s.,;!?]+')
tokens = pattern.split(sentence)
tokens = [x for x in tokens if x and x not in '-\t\n.,;!?']

list(ngrams(tokens, 2))
list(ngrams(tokens, 3))
two_grams = list(ngrams(tokens, 2))
[" ".join(x) for x in two_grams]


#########################################################################################
stop_words = ["a", "an", "the", "on", "of", "off", "this", "is"]
tokens = ["the", "house", "is", "on", "fire"]
tokens_without_stopwords = [x for x in tokens if x not in stop_words]


#########################################################################################
stop_words = nltk.corpus.stopwords.words("english")
len(stop_words)
print(stop_words[:7])
print([sw for sw in stop_words if len(sw) == 1])


#########################################################################################























