#########################################################################################
# Chapter 3
#########################################################################################

from nltk.tokenize import TreebankWordTokenizer
from collections import Counter
from collections import OrderedDict
from nlpia.data.loader import kite_text, kite_history
from nltk.corpus import brown
import nltk
import copy
import math
nltk.download('stopwords', quiet=True)
nltk.download('brown')

#########################################################################################
sentence = "The faster Harry go to the store, the faster Harry," \
           "the faster, would get home."
tokenizer = TreebankWordTokenizer()
tokens = tokenizer.tokenize(sentence.lower())
print(tokens)

bags_of_words = Counter(tokens)
print(bags_of_words)

print(bags_of_words.most_common(4))

times_harry_appears = bags_of_words['harry']
num_unique_words = len(bags_of_words)
tf = times_harry_appears/num_unique_words
print(round(tf, 4))

#########################################################################################
tokenizer = TreebankWordTokenizer()
tokens = tokenizer.tokenize(kite_text.lower())
token_counts = Counter(tokens)
print(token_counts)

#########################################################################################
stopwords = nltk.corpus.stopwords.words('english')
tokens = [x for x in tokens if x not in stopwords]
kite_counts = Counter(tokens)
print(kite_counts)

#########################################################################################
document_vector = []
doc_length = len(tokens)
for key, value in kite_counts.most_common():
    document_vector.append(value/doc_length)
print(document_vector)

#########################################################################################
docs = ["The faster Harry got to the store, the faster and faster "
        "Harry would get home."]
docs.append("Harry is hairy and faster than Jill.")
docs.append("Jill is not as hairy as Harry")

doc_tokens = []
for doc in docs:
    doc_tokens += [sorted(tokenizer.tokenize(doc.lower()))]
print(len(doc_tokens[0]))

all_doc_tokens = sum(doc_tokens, [])
len(all_doc_tokens)

lexicon = sorted(set(all_doc_tokens))
len(lexicon)
print(lexicon)

zero_vector = OrderedDict((token, 0) for token in lexicon)
print(zero_vector)

doc_vectors = []
for doc in docs:
    vec = copy.copy(zero_vector)
    tokens = tokenizer.tokenize(doc.lower())
    token_counts = Counter(tokens)
    for key, value in token_counts.items():
        vec[key] = value/len(lexicon)
    doc_vectors.append(vec)

#########################################################################################


def cosine_sim(vec1, vec2):
    """
    Function to calculate cosine similarity
    :param vec1: Vector 1
    :param vec2: Vector 2
    :return: cosine similarity score
    """
    vec1 = [val for val in vec1.values()]
    vec2 = [val for val in vec2.values()]

    dot_prod = 0
    for i, v in enumerate(vec1):
        dot_prod += v*vec2[i]

    mag_1 = math.sqrt(sum([x**2 for x in vec1]))
    mag_2 = math.sqrt(sum([x**2 for x in vec2]))

    return dot_prod/(mag_1*mag_2)


#########################################################################################
print(brown.words()[:10])
print(brown.tagged_words()[:5])
print(len(brown.words()))

puncs = {',', '.', '--', '-', '!', '?', ':', ';', '``', "''", '(', ')', '[', ']'}
word_list = (x.lower() for x in brown.words() if x not in puncs)
token_counts = Counter(word_list)
print(token_counts.most_common(20))

#########################################################################################
kite_intro = kite_text.lower()
intro_tokens = tokenizer.tokenize(kite_intro)
kite_history = kite_history.lower()
history_tokens = tokenizer.tokenize(kite_history)
intro_total = len(intro_tokens)
print(intro_total)
history_total = len(history_tokens)
print(history_total)

intro_tf = {}
history_tf = {}
intro_counts = Counter(intro_tokens)
intro_tf['kite'] = intro_counts['kite']/intro_total
history_counts = Counter(history_tokens)
history_tf['kite'] = history_counts['kites']/history_total
print(f"Term frequency of kite in intro is: {intro_tf['kite']}")
print(f"Term frequency of kite in history is: {history_tf['kite']}")

intro_tf['and'] = intro_counts['and']/intro_total
history_tf['and'] = history_counts['and']/history_total
print(f"Term frequency of and in intro is: {intro_tf['and']}")
print(f"Term frequency of and in history is: {history_tf['and']}")

num_doc_containing_and = 0
num_doc_containing_kite = 0
num_doc_containing_china = 0
for doc in [intro_tokens, history_tokens]:
    if 'and' in doc:
        num_doc_containing_and += 1
for doc in [intro_tokens, history_tokens]:
    if 'kite' in doc:
        num_doc_containing_kite += 1
for doc in [intro_tokens, history_tokens]:
    if 'china' in doc:
        num_doc_containing_china += 1

intro_tf['china'] = intro_counts['china']/intro_total
history_tf['china'] = history_counts['china']/history_total

num_docs = 2
intro_idf = {}
history_idf = {}

intro_idf['and'] = num_docs/num_doc_containing_and
history_idf['and'] = num_docs/num_doc_containing_and
intro_idf['kite'] = num_docs/num_doc_containing_kite
history_idf['kite'] = num_docs/num_doc_containing_kite
intro_idf['china'] = num_docs/num_doc_containing_china
history_idf['china'] = num_docs/num_doc_containing_china

intro_tfidf = {}
history_tfidf = {}
intro_tfidf['and'] = intro_tf['and']*intro_idf['and']
intro_tfidf['kite'] = intro_tf['kite']*intro_idf['kite']
intro_tfidf['china'] = intro_tf['china']*intro_idf['china']
history_tfidf['and'] = history_tf['and']*history_idf['and']
history_tfidf['kite'] = history_tf['kite']*history_idf['kite']
history_tfidf['china'] = history_tf['china']*history_idf['china']

#########################################################################################
document_tfidf_vectors = []
for doc in docs:
    vec = copy.copy(zero_vector)
    tokens = tokenizer.tokenize(doc.lower())
    token_counts = Counter(tokens)

    for key, value in token_counts.items():
        docs_containing_key = 0
        for _doc in docs:
            if key in _doc:
                docs_containing_key += 1
        tf = value/len(lexicon)
        if docs_containing_key:
            idf = len(docs)/docs_containing_key
        else:
            idf = 0
        vec[key] = tf*idf

    document_tfidf_vectors.append(vec)

























