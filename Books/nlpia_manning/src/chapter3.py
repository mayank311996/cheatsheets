#########################################################################################
# Chapter 3
#########################################################################################

from nltk.tokenize import TreebankWordTokenizer
from collections import Counter
from collections import OrderedDict
from nlpia.data.loader import kite_text
import nltk
import copy
nltk.download('stopwords', quiet=True)

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












