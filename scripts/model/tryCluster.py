from sklearn.cluster import AffinityPropagation, KMeans
from gensim.models import Word2Vec
import pickle
import numpy as np
from scripts.prepro.corpusManagement import getUnderstander
import random
import pandas as pd
from itertools import chain

file = open('pickles/fullytreated_corpus.pkl', 'rb')
sentences = pickle.load(file)
vector_size=200
model = Word2Vec(sentences=sentences, vector_size=vector_size, window=5, alpha=0.75) 

file = open('working_pickles/meta.pkl', 'rb')
meta = pickle.load(file)

# first, we pool the embeddings of words within a paragraph together. our baseline pooling function is to average across a vector element

para_array = []
considered_vocab = model.wv.key_to_index.keys()
sentences = [ [word for word in sentence if word in considered_vocab] for sentence in sentences ]# also remove from consideration if the paragraph is so short as to only include very rare words that barely appear

understander = getUnderstander(garble = sentences, legible = meta)

sentences = [sentence for sentence in sentences if (sentence != [])]# there are paragraphs which are now completely empty because we strip it of punctuation, urls, numbers etc previously. 
    # so I decide to just remove those sentences because it will result in zero vectors which could affect modeling


for id, sentence in enumerate(sentences):

    para_vector = np.average(model.wv[sentence], axis=0)
    addition = para_vector.tolist()
    para_array.append(addition)

    print(id / len(sentences))

#sampled_array = random.sample(para_array, 5000)

clusterModel = KMeans(n_clusters=8, random_state=99).fit(para_array)

labels = pd.Series(clusterModel.labels_)
garble = pd.Series(sentences)
legible = pd.Series([understander[tuple(sentence)] for sentence in sentences])

readable = pd.concat([labels, garble, legible], axis=1)
readable.columns = ['labels', 'garble', 'legible']

# elbow method
elbow = []
for i in [10, 15, 20, 25, 30]:
    clusterModel = KMeans(n_clusters=i, random_state=99).fit(para_array)
    elbow.append([i, clusterModel.inertia_])

for i in [35, 40, 45, 50]:
    clusterModel = KMeans(n_clusters=i, random_state=99).fit(para_array)
    elbow.append([i, clusterModel.inertia_])

import matplotlib.pyplot as plt

plt.plot(elbow)
plt.show()
pd.DataFrame(elbow)

# for a cluster of 50

clusterModel = KMeans(n_clusters=50, random_state=99).fit(para_array)

labels = pd.Series(clusterModel.labels_)
garble = pd.Series(sentences)
legible = pd.Series([understander[tuple(sentence)] for sentence in sentences])

readable = pd.concat([labels, garble, legible], axis=1)
readable.columns = ['labels', 'garble', 'legible']

readable[readable['labels'] == 44][100:150]