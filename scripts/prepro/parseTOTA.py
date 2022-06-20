import xml.etree.ElementTree as ET
import pandas as pd
import os

df = pd.DataFrame()
lol=[]

# Passing the path of the
# xml document to enable the
# parsing process

def getelementfromHead(elementHead, elementName):
    items = elementHead.items()
    titleList = [item[0] for item in items]
    whereelement = titleList.index(elementName)
    returntext = items[whereelement][1]

    return returntext

for xml in os.listdir('tota'):
    tree = ET.parse('tota/'+xml)
    root = tree.getroot()
    meta = root[0]

    """
    do a quick check of the meta: if it is non-english, we do not want it
    """
    lang = meta.findall('language')[0]
    if lang.text != 'en':
        continue

    body = root[1]
    for chapter in body:
        if 'name' not in chapter.keys():
            continue # we are not interested if a chapter doesn't have a name. for now - it's not useful for determining clause categories
        else:
            chapter_name = getelementfromHead(chapter, 'name')

        for article in chapter:
            if 'name' not in article.keys():
                continue # we are not interested if an article doesn't have a name. for now - it's not useful for determining clause categories
            article_name = getelementfromHead(article, 'name')
            article_text = article.text
            
            lol.append([chapter_name, article_name, article_text])

df = pd.DataFrame(lol)

df.columns=['chapter', 'article', 'text']



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
article_lower = [article.lower() for article in df['article']]
clause_lower = [clause.lower() for clause in df['text']]
article_embed = model.encode(article_lower, show_progress_bar=True)
clause_embed = model.encode(clause_lower, show_progress_bar=True)

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

"""
Strategy #2: now that we have our chapter clusters, we want to cluster our article titles
- We use DBSCAN so that we don't have to determine the number of clusters
- minimum number of samples = 2, to catch even irregular clause types
- eps = 0.8, so that words with a small edit distance can still be lumped into the same category
- for now, we discard noisy samples
"""

from sklearn.cluster import DBSCAN

newdf = pd.DataFrame()

for chapter in pd.unique(df['chapter_cat']):

    area = df[df['chapter_cat'] == chapter].reset_index().drop('index',axis=1)

    relevant_embed = article_embed[df['chapter_cat'] == chapter]

    clustermodel=DBSCAN(min_samples=2, eps=0.8)
    clusters = clustermodel.fit_predict(relevant_embed)

    area = pd.concat([area, pd.Series(clusters)],axis=1)
    area=area.rename(columns={0:'article_cat'})

    area = area[area['article_cat']!=-1]

    newdf=newdf.append(area)