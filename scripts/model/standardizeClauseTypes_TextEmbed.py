import pandas as pd

import sys
sys.path.append("scripts/model/")
from cleanandTransform import cleanandTransform

df=pd.read_csv('core_excels/articlesAssigned.csv')
df=df[['chapter', 'chapter_cat', 'article', 'clauseno', 'text', 'fta_name', 'rta_id', 'toa', 'parties', 'dif','isSubstantive_predict', ]]



################
"""
FIX THE UKFTA/DEA CHAPTER NAMES THANK U
"""


"""
- our present goal: unsupervised classification of clauses
- answer the question: what kind of clause type is a particular clause?
- brings up the question of what counts exactly as a defined clause type
- one of two methods we can use to represent an FTA text
- final goal being the identification of specific clause types eith economic impact
"""

# create embeddings for our texts

artcat = cleanandTransform(filters = [], transformer_package='all-MiniLM-L12-v2')
#try minilm first

arts = df.text.apply(lambda x: str(x))

artcat.init_text(arts)

artcat.set_filters(['to_lower', 'remove_url','remove_special_character', 'normalize_unicode', 'remove_punctuation', 'remove_number', 'remove_whitespace'])

artcat.process_text()

artcat.transform_text()
art_embed = artcat.current_embedding

from sklearn.cluster import DBSCAN, AffinityPropagation
from sklearn.decomposition import PCA



newdf = pd.DataFrame()
#0.85
for chapter in pd.unique(df['chapter_cat']):

    area = df[df['chapter_cat'] == chapter].reset_index().drop('index',axis=1)

    relevant_embed = art_embed[df['chapter_cat'] == chapter]
    pca = PCA(n_components=25)

    decomposed = pca.fit_transform(relevant_embed)
    clustermodel = AffinityPropagation()
    clusters = clustermodel.fit_predict(decomposed)

    area = pd.concat([area, pd.Series(clusters)],axis=1)
    area=area.rename(columns={0:'article_cat'})

    area = area[area['article_cat']!=-1]

    newdf=newdf.append(area)

newdf=newdf.rename(columns={0:'article_cat'})
newdf=newdf.reset_index()

"""
Give a verbose name to each article category. Naively, we simply pick whatever happens to be the most occuring phrase and assign the category as that
"""

keyer = newdf.groupby(['chapter_cat', 'article_cat'])['article'].agg( lambda x: pd.Series.mode(x)[0])

newdf=newdf.merge(keyer, how='left', on=['chapter_cat', 'article_cat'])

newdf=newdf.rename({
    'article_x':'article',
    'article_y':'article_label'
},axis=1)

newdf.to_csv('test.csv')
newdf.to_csv("core_excels/totaPlus_textembed.csv")





#### try pca to 2d and visualize

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

decomposed = pca.fit_transform(relevant_embed)

import plotly.express as px
fig = px.scatter(x=decomposed[:,0], y=decomposed[:,1])
fig.show()