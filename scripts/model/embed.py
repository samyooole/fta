from gensim.models import Word2Vec
import pickle

file = open('pickles/fullytreated_corpus.pkl', 'rb')
sentences = pickle.load(file)

"""
to study a bit deeper on how vector size affects the sensibility of the embedding model
"""

model = Word2Vec(sentences=sentences, vector_size=200, window=5) 