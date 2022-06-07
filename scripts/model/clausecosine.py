from sentence_transformers import SentenceTransformer
import datasets
import pickle
from random import sample, seed
from scipy.spatial.distance import cosine
from numpy import argmax
import pandas as pd
import numpy as np

df = pd.read_csv('fta_clauses_list.csv')

df.verbose.fillna(df.clause, inplace=True)

dsdf = datasets.Dataset.from_pandas(df)
dsdf = dsdf.rename_column('verbose', 'text')

model = SentenceTransformer('all-mpnet-base-v2')
embeddings = model.encode(dsdf['text'])

clausecat_embed = embeddings

file = open('pickles/fullytreated_corpus.pkl', 'rb')
text = pickle.load(file)
seed(1965)
#sample_text = text
#corpus_embed = model.encode(sample_text)

sample_text = text[0:50]
corpus_embed = model.encode(sample_text)

closestcats=[]

for clause in corpus_embed:
    scores=[]
    for clausecat in clausecat_embed:
        score = 1 - cosine(clause, clausecat) # scipy's cosine gives distance, so we must have 1 - that to get similarity
        scores.append(score)
    index = argmax(scores)
    closestcat = dsdf['text'][index]
    closestcats.append(closestcat)

import pandas as pd
df = pd.concat([pd.Series(sample_text[0:len(closestcats)-1]), pd.Series(closestcats)], axis=1)

df.to_csv('sample_clause_cosinesim.csv')
