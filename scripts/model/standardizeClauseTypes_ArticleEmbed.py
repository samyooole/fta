import pandas as pd

import sys
sys.path.append("scripts/model/")
from cleanandTransform import cleanandTransform

df=pd.read_csv('core_excels/totaPlus_textembed.csv')
df=df[['chapter', 'chapter_cat', 'article', 'clauseno', 'text', 'fta_name', 'rta_id', 'toa', 'parties', 'dif','isSubstantive_predict', 'article_cat', 'article_label']]

df=df.rename({'article_cat':'clausetype_num', 'article_label':'clausetype_alph'},axis=1)

################
"""
FIX THE UKFTA/DEA CHAPTER NAMES THANK U
"""


"""
Strategy #2: now that we have our chapter clusters, we want to cluster our article titles
- We use DBSCAN so that we don't have to determine the number of clusters
- minimum number of samples = 2, to catch even irregular clause types
- eps = 0.8, so that words with a small edit distance can still be lumped into the same category
- for now, we discard noisy samples
"""

# create embeddings for our raw article titles

artcat = cleanandTransform(filters = []) #allmpnet being used

from sklearn.cluster import OPTICS, DBSCAN, AffinityPropagation

newdf = pd.DataFrame()
#0.85
for chapter in pd.unique(df['chapter_cat']):

    area = df[df['chapter_cat'] == chapter].reset_index().drop('index',axis=1)

    arts = area.article.astype('string')
    arts = pd.unique(area.article)

    artcat.init_text(arts)

    artcat.transform_text()
    relevant_embed = artcat.current_embedding

    #clustermodel=DBSCAN(min_samples=2, eps=0.83)
    clustermodel = AffinityPropagation()
    clusters = clustermodel.fit_predict(relevant_embed)

    key1 = pd.concat([pd.Series(arts), pd.Series(clusters)], axis=1)
    key1.columns = ['article', 'art_label']

    area = area.merge(key1, how='left', left_on='article', right_on = 'article')

    area=area.rename({'art_label':'article_cat'},axis=1)

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

newdf.to_csv("core_excels/totaPlusArtNameEmbed.csv")


"""
some cleanup"""

newdf = newdf[['chapter', 'article', 'clauseno', 'text', 'isSubstantive_predict', 'chapter_cat', 'article_label', 'article_cat', 'clausetype_alph', 'clausetype_num', 'fta_name', 'rta_id', 'parties', 'dif']]

newdf = newdf.rename({'article_cat' : 'article_cat_no' , 'article_label' : 'article_cat', 'clausetype_alph' : 'clauseSem', 'clausetype_num' : 'clauseSem_no'}, axis=1)

newdf.to_csv("core_excels/totaPlusArtNameEmbed.csv")

"""
a quick note. i am leaving the article clusterings as it is, even though they are a bit unaggressive, ie. 'definition' and 'definitions and scope' are two different clusters. although my personal inclination is that these two should be different clusters, there is an argument to be made that they should be separate clusters. i will leave further categorization till later, if there happens to be a use case for 


edit> now we try to combine similarly worded clusters together
"""

newdf = pd.read_csv('core_excels/totaPlus.csv')

chaps = pd.unique(newdf.chapter_cat)

aggcat = cleanandTransform(filters = [])
clusterModel = OPTICS(min_samples=2, metric='cosine')


finaldf = pd.DataFrame()
for chap in chaps:
    smalldf = newdf[newdf['chapter_cat'] == chap]

    arts = pd.unique(smalldf.article_cat)

    arts = [string.replace("-", " ") for string in arts]

    aggcat.init_text(arts)
    aggcat.transform_text()

    labels = clusterModel.fit_predict(aggcat.current_embedding)

    clusterdf = pd.concat([pd.Series(arts, name='arts'), pd.Series(labels, name='labels')], axis=1)

    clusterdf['newart'] = clusterdf.apply(lambda x: x['arts'] if x['labels'] == -1 else None, axis=1)

    s = clusterdf[clusterdf['labels'] != -1].arts.str.len().sort_values().index

    concisedf = clusterdf[clusterdf['labels'] != -1].reindex(s)

    concisedf = concisedf.drop_duplicates('labels')

    concisedf=concisedf[['arts', 'labels']].rename({'arts':'newarts'},axis=1)

    clusterdf = clusterdf.merge(concisedf, how='left',)
    clusterdf['finalnewart'] = clusterdf.apply(lambda x: x['newart'] if pd.notnull(x['newart']) else x['newarts'],axis=1)

    clusterdf=clusterdf.drop(['labels', 'newart', 'newarts', ],axis=1)

    newsmalldf = smalldf.merge(clusterdf, how='left', left_on='article_cat', right_on='arts').drop(['Unnamed: 0', 'article_cat', 'article_cat_no'], axis=1)

    finaldf = pd.concat([finaldf, newsmalldf], axis=0)


   




"""
try strategy #2, but instead with levenshtein distances as a metric
"""
import numpy as np
from sklearn.cluster import AffinityPropagation
import distance
    
words = np.asarray(artcat.current_text) #So that indexing with a list will work
lev_similarity = -1*np.array([[distance.levenshtein(w1,w2) for w1 in words] for w2 in words])

# rmb to pickle this metric later

affprop = AffinityPropagation(affinity="precomputed", damping=0.5)
affprop.fit(lev_similarity)
for cluster_id in np.unique(affprop.labels_):
    exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
    cluster = np.unique(words[np.nonzero(affprop.labels_==cluster_id)])
    cluster_str = ", ".join(cluster)
    print(" - *%s:* %s" % (exemplar, cluster_str))


