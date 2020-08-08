#########################################################################################
# Chapter 6
#########################################################################################

import os
import seaborn
import numpy as np
import pandas as pd
import multiprocessing
from nlpia.book.examples.ch06_nessvectors import *
from nlpia.data.loaders import get_data
# this conflicts with later import
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import Word2Vec
from gensim.models.fasttext import FastText
from gensim.models.word2vec import KeyedVectors
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from nlpia.plots import offline_plotly_scatter_bubble
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim.utils import simple_preprocess

#########################################################################################
print(nessvector("Marie_Curie").round(2))

#########################################################################################
word_vectors = get_data('word2vec')

#########################################################################################
word_vectors = KeyedVectors.load_word2vec_format(
    '/path/to/GoogleNews-vectors-negative300.bin.gz',
    binary=True
)
word_vectors = KeyedVectors.load_word2vec_format(
    '/path/to/GoogleNews-vectors-negative300.bin.gz',
    binary=True,
    limit=200000
)

#########################################################################################
word_vectors.most_similar(positive=['cooking', 'potatoes'], topn=5)
word_vectors.most_similar(positive=['germany', 'france'], topn=5)

word_vectors.doesnt_match("potatoes milk cake computer".split())

word_vectors.most_similar(
    positive=['king', 'woman'],
    negative=['man'],
    topn=2
)

word_vectors.similarity('princess', 'queen')

print(word_vectors['phone'])

#########################################################################################
token_list = []  # Fill token list with tokens of sentences in a corpus

NUM_FEATURES = 300
MIN_WORD_COUNT = 3
NUM_WORKERS = 2  # num_workers = multiprocessing.cpu_count(),
# import multiprocessing first
WINDOW_SIZE = 6
SUBSAMPLING = 1e-3

model = Word2Vec(
    token_list,
    workers=NUM_WORKERS,
    size=NUM_FEATURES,
    min_count=MIN_WORD_COUNT,
    window=WINDOW_SIZE,
    sample=SUBSAMPLING
)

model.init_sims(replace=True)

model_name = "my_domain_specific_word2vec_model"
model.save(model_name)

model = Word2Vec.load(model_name)
model.most_similar('radiology')

#########################################################################################
MODEL_PATH = "/path/"
ft_model = FastText.load_fasttext_format(
    model_file=MODEL_PATH
)
ft_model.most_similar("soccer")

#########################################################################################
wv = get_data('word2vec')
print(len(wv.vocab))

vocab = pd.Series(wv.vocab)
print(vocab.iloc[1000000:1000006])

print(wv["Illini"])

print(np.linalg.norm(wv["Illinois"] - wv["Illini"]))
cos_similarity = np.dot(wv["Illinois"], wv["Illini"]) / (
    np.linalg.norm(wv["Illinois"]) *
    np.linalg.norm(wv["Illini"])
)
print(cos_similarity)
print(1-cos_similarity)

#########################################################################################
cities = get_data("cities")
print(cities.head(1).T)

us = cities[
    (cities.country_code == "US")
    & (cities.admin1_code.notnull())
].copy()
states = pd.read_csv(
    "http://www.fonz.net/blog/wp-content/uploads/2008/04/states.csv"
)
states = dict(zip(states.Abbreviation, states.State))
us['city'] = us.name.copy()
us['st'] = us.admin1_code.copy()
us['state'] = us.st.map(states)
us[us.columns[-3:]].head()

vocab = pd.np.concatenate([us.city, us.st, us.state])
vocab = np.array([word for word in vocab if word in wv.wv])
print(vocab[:5])

city_plus_state = []
for c, state, st in zip(us.city, us.state, us.st):
    if c not in vocab:
        continue
    row = []
    if state in vocab:
        row.extend(wv[c] + wv[state])
    else:
        row.extend(wv[c] + wv[st])
    city_plus_state.append(row)
us_300D = pd.DataFrame(city_plus_state)

pca = PCA(n_components=2)
us_300D = get_data("cities_us_wordvectors")
us_2D = pca.fit_transform(us_300D.iloc[:, :300])

#########################################################################################
df = get_data("cities_us_wordvectors_pca2_meta")
html = offline_plotly_scatter_bubble(
    df.sort_values('population', ascending=False)[:350].copy()
    .sort_values('population'),
    filename='plotly_scatter_bubble.html',
    x='x',
    y='y',
    size_col='population',
    text_col='name',
    category_col='timezone',
    xscale=None,
    yscale=None,
    layout={},
    marker={'sizeref': 3000}
)

#########################################################################################
num_cores = multiprocessing.cpu_count()
corpus = [
    'This is the first document',
    "this is the second document"
]
training_corpus = []
for i, text in enumerate(corpus):
    tagged_doc = TaggedDocument(
        simple_preprocess(text),
        [i]
    )
    training_corpus.append(tagged_doc)

model = Doc2Vec(
    size=100,
    min_count=2,
    workers=num_cores,
    iter=10
)
model.build_vocab(training_corpus)
model.train(
    training_corpus,
    total_examples=model.corpus_count,
    epochs=model.iter
)

model.infer_vector(
    simple_preprocess("This is an unseen doc"),
    steps=10
)





