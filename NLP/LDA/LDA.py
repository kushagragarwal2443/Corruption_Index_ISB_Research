import csv
import pandas as pd;
import numpy as np;
import scipy as sp;
import sklearn;
import sys;
from nltk.corpus import stopwords;
import nltk;
from gensim.models import ldamodel
import gensim.corpora;
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer;
from sklearn.decomposition import NMF;
from sklearn.preprocessing import normalize;
import pickle

def get_lda_topics(model, num_topics):
    word_dict = {}
    for i in range(num_topics):
        words = model.show_topic(i, topn = 20)
        word_dict[i] = [i[0] for i in words]
    # return pd.DataFrame(word_dict)
    return word_dict


data = pd.read_csv('./y1819.csv', error_bad_lines=False)
datacolumn = "Text"
# print(data)

data_text = data[[datacolumn]]
# print(data_text)

data_text = data_text.astype('str')

for idx in range(len(data_text)):
    
    #go through each word in each data_text row, remove stopwords, and set them on the index.
    data_text.iloc[idx][datacolumn] = [word for word in data_text.iloc[idx][datacolumn].split(' ') if word not in stopwords.words()]
    
    #print logs to monitor output
    sys.stdout.write('\rc = ' + str(idx+1) + ' / ' + str(len(data_text)))


print()

train_headlines = [value[0] for value in data_text.iloc[0:].values]
# print(train_headlines)
num_topics= 25
id2word = gensim.corpora.Dictionary(train_headlines)
corpus = [id2word.doc2bow(text) for text in train_headlines]
lda = ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics)

filehander = open("result.txt",'w')
dictionary = (get_lda_topics(lda, num_topics))
for i in range(num_topics):
    filehander.write("\n\nTopic {}\n".format(i+1))
    filehander.write(str(dictionary[i]))
    if("corruption" in dictionary[i]):
        print("Found in Topic {}".format(i+1))
    
