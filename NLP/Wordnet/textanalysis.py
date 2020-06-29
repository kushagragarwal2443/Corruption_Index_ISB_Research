import nltk
from nltk.corpus import wordnet as wn


corruption_synsets = wn.synsets('sight', 'n')
print(corruption_synsets)

for corr in corruption_synsets:
    print("LEMMAS:",corr.lemmas())
    print("DEFINITION:",corr.definition())
    print("HYPERNYMS:", corr.hypernyms())
    print("HYPONYMS:", corr.hyponyms())
    print()
