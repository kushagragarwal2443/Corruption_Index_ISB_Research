import pandas as pd
import re
import nltk
import gensim
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from gensim.models import Word2Vec


word1 = "corruption"
word2 = "bribe"

# Tokenize and lemmatize
def lemmatize_stemming(text):
    stemmer = SnowballStemmer("english")
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

# USE FOR CSV
data = pd.read_csv('../y1819.csv', error_bad_lines=False)
datacolumn = "Text"
data_text = data[[datacolumn]]
articles =""
count =0
for line in data_text[datacolumn]:
    line = str(line)
    # Condition used is the following:
    # if (corrupt or bribe) and government
    if(("corruption" in line or "corrupt" in line or "bribe" in line or "bribery" in line) and "government" in line):
        articles= articles + line
        count+=1
print(count)

# USE FOR TXT
# articles = ""
# data = open("randomarticle", "r")
# for line in data:
#     articles = articles + str(line)

# This converts article to lower case and removes all other characters with a space
processed_article = articles.lower()
processed_article = re.sub('[^a-zA-Z]', ' ', processed_article )
processed_article = re.sub(r'\s+', ' ', processed_article)

# # USE FOR UNLEMMATIZED
# all_sentences = nltk.sent_tokenize(processed_article)
# all_words = [nltk.word_tokenize(sent) for sent in all_sentences]
# from nltk.corpus import stopwords
# for i in range(len(all_words)):
#     all_words[i] = [w for w in all_words[i] if w not in stopwords.words('english')]

# USE FOR LEMMATIZED
all_words = preprocess(processed_article)
random_list = []
random_list.append(all_words)
all_words = random_list
word1 = str((preprocess(word1))[0])
word2 = str((preprocess(word2))[0])

word2vec = Word2Vec(all_words, min_count=1)

v1 = word2vec.wv[word1]
v2 = word2vec.wv[word2]

# Find most similar words to corruption and bribe
sim_words2 = word2vec.wv.most_similar(word2)
sim_words1 = word2vec.wv.most_similar(word1)

# Find cosine distance between corruption and bribe
similarity= word2vec.similarity(word1, word2)

print("Word 1: {}".format(word1))
print("Word 2: {}\n".format(word2))

print("Word 1 vector\n")
print(v1)
print("\n\n\n")

print("Word 2 vector\n")
print(v2)
print("\n\n\n")

print("Words similar to Word 1\n")
print(sim_words1)
print("\n\n\n")

print("Words similar to Word 2\n")
print(sim_words2)
print("\n\n\n")

print("Similarity between the two\n")
print(similarity)
