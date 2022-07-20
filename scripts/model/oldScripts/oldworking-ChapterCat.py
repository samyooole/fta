
"""
AN OLD working script that allowed us to figure chapter categories.

after this k means classification performed on tota, i manually reclassified some areas according to my personal judgement - that is not reflected here. the final results are reflected in archive/working_excels/totachapters
"""



df = pd.read_csv('newtota.csv')




# first, check the unique values of chapter
len(pd.unique(df['chapter']))
unique_chapters = pd.unique(df['chapter'])
unique_chapters = [chapter.lower() for chapter in unique_chapters] # convert all words to lowercase
unique_chapters_proper = pd.unique(df['chapter'])

"""
we want to take all the varied chapters, and cluster them into around 10-20 proper groups

Strategy #1: create sentence embeddings and use clustering to get them into groups, possibly DBSCAN
"""

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-mpnet-base-v2')
embed = model.encode(unique_chapters, show_progress_bar=True)


"""
some other legwork: encode all the relevant embeddings
"""
import re
article_lower = [article.lower() for article in df['article']]
clause_lower = [clause.lower() for clause in df['text']]

mid = [re.split(r"\. *?\d+\. *?", item, flags=re.DOTALL | re.I) for item in clause_lower]
mid = [ [item.replace("\n", " ") for item in intermed] for intermed in mid]
mid = [ [item.replace("\t", " ") for item in intermed] for intermed in mid]

df['splitted'] = mid
df=df.explode('splitted')
df=df.reset_index()

article_embed = model.encode(article_lower, show_progress_bar=True)
clause_embed = model.encode(df['splitted'], show_progress_bar=True)

import pickle
with open('pickles/tota_art_embed.pkl', 'wb') as file:
    pickle.dump(article_embed, file)
with open('pickles/tota_clause_embed.pkl', 'wb') as file:
    pickle.dump(clause_embed, file)


from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

"""
elbow method: find ideal number of clusters"""

clustermodel = KMeans()
visualizer = KElbowVisualizer(clustermodel, k=(30,50))
visualizer.fit(embed)
visualizer.show()

clustermodel = KMeans(n_clusters=41)
clusters = clustermodel.fit_predict(embed)

clusterdf=pd.concat([pd.Series(unique_chapters_proper), pd.Series(clusters)], axis=1)
clusterdf.columns = ['chapter', 'cluster']

clusterdf.to_csv('clusterdf.csv') # write out as csv, manually re-classify

"""
reload back in, map accordingly"""

totachapters=pd.read_csv('important_excels/tota_chapters.csv')
chapterdict=pd.read_csv('important_excels/chapter_dict.csv')


totadict=totachapters.merge(chapterdict, how='left', left_on='cluster', right_on='chapter_label')

df=df.merge(totadict, how='left')
df=df.drop(['cluster','chapter_label'],axis=1)