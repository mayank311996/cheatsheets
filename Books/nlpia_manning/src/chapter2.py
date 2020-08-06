#########################################################################################
# Chapter 2
#########################################################################################

import numpy as np
import pandas as pd
import re
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import TreebankWordTokenizer
from nltk.util import ngrams
import nltk
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS \
    as sklearn_stop_words
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nlpia.data.loaders import get_data
from nltk.tokenize import casual_tokenize
from sklearn.naive_bayes import MultinomialNB
nltk.download("wordnet")
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
len(sklearn_stop_words)
len(stop_words)
len(stop_words.union(sklearn_stop_words))
len(stop_words.intersection(sklearn_stop_words))

#########################################################################################
tokens = ["House", "Visitor", "Center"]
normalized_tokens = [x.lower() for x in tokens]
print(normalized_tokens)


#########################################################################################
def stem(phrase):
    return " ".join([re.findall('^(.*ss|.*?)(s)?$', word)
                     [0][0].strip("'") for word in phrase.lower().split()])


print(stem('houses'))
print(stem("Doctor House's calls"))

#########################################################################################
lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize("better")
lemmatizer.lemmatize("better", pos="a")
lemmatizer.lemmatize("good", pos="a")
lemmatizer.lemmatize("goods", pos="a")
lemmatizer.lemmatize("goods", pos="n")
lemmatizer.lemmatize("goodness", pos="n")
lemmatizer.lemmatize("best", pos="a")

#########################################################################################
sa = SentimentIntensityAnalyzer()
print(sa.lexicon)
print([(tok, score) for tok, score in sa.lexicon.items() if " " in tok])

sa.polarity_scores(
    text="Python is very readable and it's great for NLP."
)
sa.polarity_scores(
    text="Python is not a bad choice for most applications."
)

corpus = [
    "Absolutely perfect! Love it! :-) :-) :-)",
    "Horrible! Completely useless. :(",
    "It was OK. Some good and some bad things."
]
for doc in corpus:
    scores = sa.polarity_scores(doc)
    print(f"{scores['compound']}:{doc}")

#########################################################################################
movies = get_data("hutto_movies")
movies.head().round(2)
movies.describe().round(2)

pd.set_option("display.width", 75)

bags_of_words = []
for text in movies.text:
    bags_of_words.append(Counter(casual_tokenize(text)))
df_bows = pd.DataFrame.from_records(bags_of_words)
df_bows = df_bows.fillna(0).astype(int)
print(df_bows.shape)
df_bows.head()
df_bows.head()[list(bags_of_words[0].keys())]

nb = MultinomialNB()
nb = nb.fit(df_bows, movies.sentiment > 0)
movies['predicted_sentiment'] = nb.predict_proba(df_bows)*8 - 4
movies['error'] = (movies.predict_sentiment - movies.sentiment).abs()
print(movies.error.mean().round(1))
movies['sentiment_ispositive'] = (movies.sentiment > 0).astype(int)
movies['predicted_ispositive'] = (movies.predict_sentiment > 0).astype(int)
movies["sentiment predicted_sentiment sentiment_ispositive" \
       " predicted_ispositive".split()].head(8)

#########################################################################################





















