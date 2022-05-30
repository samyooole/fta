from sklearn.cluster import AffinityPropagation
from gensim.models import Word2Vec
import pickle
import numpy as np

file = open('pickles/fullytreated_corpus.pkl', 'rb')
sentences = pickle.load(file)
vector_size=200
model = Word2Vec(sentences=sentences, vector_size=vector_size, window=5) 

para_array = np.empty(shape = vector_size)

for sentence in sentences:
    para_vector = np.average(model.wv[sentence], axis=0)
    para_array = np.insert(para_array, para_vector, axis=1)

np.average(model.wv['sanit', 'export'], axis=0)
model.wv['sanit', 'export'].avera

# first, we must pool every paragraph to understand 

df = model.wv.vectors

clusterModel = AffinityPropagation(random_state=5, verbose=True).fit(df)