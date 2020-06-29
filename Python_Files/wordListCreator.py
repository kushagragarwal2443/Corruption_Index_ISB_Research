
import os
import pandas as pd
import nltk
import gensim
from gensim import corpora, models , similarities
from gensim.models import KeyedVectors
import re

from nltk.stem import WordNetLemmatizer, SnowballStemmer

# Preprocess article
# Stemming and Lemmatization
def lemmatize_stemming(text):
    stemmer = SnowballStemmer("english")
    # Using verb form of the word
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        # Take only those words which appear atleast thrice in the data
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


# List was created using the 4 years data 
df = pd.read_csv(("../The_hindu_scrapped/y18.csv"))
df1 = pd.read_csv(("../The_hindu_scrapped/y17.csv"))
df2 = pd.read_csv(("../The_hindu_scrapped/y16.csv"))
df3 = pd.read_csv(("../The_hindu_scrapped/y15.csv"))
df = df.append(df1,ignore_index = True)
df = df.append(df2,ignore_index = True)
df = df.append(df3,ignore_index = True)


df = df.dropna()
sentences = []
df = df.reset_index(drop =True)

# Iterate over all articles
for i in range(len(df["Text"])):
    
    # Convert article to lower case and remove all other characters
    article = str(df["Text"][i]).lower()
    article = re.sub('[^a-zA-Z]', ' ', article )
    
    # Lemmatize and Stem the resulting article
    article = preprocess(article)
    sentences.append(article)
    if(i%1000==0):
        print (i)

# Using word2vec ML architechture on out sentences list
model = gensim.models.Word2Vec(sentences,min_count = 1 , size = 64)
# size = 64 means each word is mapped to a 64 dimensional space

# finds 30 words with least cosine distance from the word corrupt
word_weights = model.most_similar(positive = ["corrupt"],topn = 30)

words = []
weights = []

# create list of words and their associated words
for i in range(len(word_weights)):
    words.append(word_weights[i][0])
    weights.append(word_weights[i][1])
print(words)
print(weights)
