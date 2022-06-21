import pandas as pd
import pickle
from scipy.spatial.distance import cosine
from numpy import argmax

totadf = pd.read_csv("pickles/totadf.csv")
with open('pickles/tota_clause_embed.pkl', 'rb') as f:
    tota_embed = pickle.load(f)

sgftadf = pd.read_csv("pickles/text.csv")
with open('pickles/sgfta_clauselevel_relevant_embeddings.pkl', 'rb') as f:
    sgfta_embed = pickle.load(f)

# for cleanliness sake, we take out those with -1 tag
kept_indices = totadf['article_cat'] != -1
totadf = totadf[kept_indices]
totadf=totadf.reset_index()
tota_embed = tota_embed[kept_indices]

closestcats=[]

for idx, clause in enumerate(sgfta_embed):
    scores=[]
    for clausecat in tota_embed:
        score = 1 - cosine(clause, clausecat) # scipy's cosine gives distance, so we must have 1 - that to get similarity
        scores.append(score)
    index = argmax(scores)
    print(totadf['text'][index])
    rel_chapter = totadf['chapter_cat'][index]
    rel_article = totadf['article_label'][index]
    closestcats.append([rel_chapter, rel_article])
    print(idx/len(sgfta_embed))




##
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-mpnet-base-v2')

artchap_indexed = totadf['chapter'] +" " + totadf['article']
artchap_unique = pd.unique(artchap_indexed)

artchap_embed = model.encode(artchap_unique, show_progress_bar=True)


closestcats=[]

for idx, clause in enumerate(sgfta_embed):
    scores=[]
    for clausecat in artchap_embed:
        score = 1 - cosine(clause, clausecat) # scipy's cosine gives distance, so we must have 1 - that to get similarity
        scores.append(score)
    index = argmax(scores)
    rel_chapter = artchap_unique[index]
    closestcats.append(rel_chapter)
    print(idx/len(sgfta_embed))

closestcatsdf=pd.DataFrame(closestcats)