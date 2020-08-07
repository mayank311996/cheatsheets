#########################################################################################
# Chapter 4
#########################################################################################

import pandas as pd
from nlpia.data.loaders import get_data
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize
from sklearn.preprocessing import MinMaxScaler
from pugnlp.stats import Confusion

#########################################################################################
pd.options.display.width = 120
sms = get_data('sms-spam')
index = [f'sms{i}{"!"*j}' for (i, j) in zip(range(len(sms)), sms.spam)]
sms = pd.DataFrame(
    sms.values,
    columns=sms.columns,
    index=index
)
sms['spam'] = sms.spam.astype(int)
print(len(sms))
print(sms.spam.sum())
print(sms.head(6))

tfidf_model = TfidfVectorizer(tokenizer=casual_tokenize)
tfidf_docs = tfidf_model.fit_transform(raw_documents=sms.text).toarray()
print(tfidf_docs.shape)

mask = sms.spam.astype(bool).values
spam_centroid = tfidf_docs[mask].mean(axis=0)
ham_centroid = tfidf_docs[~mask].mean(axis=0)
print(spam_centroid.round(2))
print(ham_centroid.round(2))

spamminess_score = tfidf_docs.dot(spam_centroid - ham_centroid)
print(spamminess_score.round(2))

sms['lda_score'] = MinMaxScaler().fit_transform(
    spamminess_score.reshape(-1,1)
)
sms['lda_predict'] = (sms.lda_score > 0.5).astype(int)
print(sms['spam lda_predict lda_score'.split()].round(2).head(6))

print((1.0 - (sms.spam - sms.lda_predict).abs().sum()/len(sms)).round(3))

Confusion(sms['spam lda_predict'.split()])






