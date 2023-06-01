"""
Chapter names may be different from each other but in fact mean the same thing

We previously did some manual labeling to unite these slightly different fields.

Can load the model as explained below

"""

import pandas as pd

import sys
sys.path.append("scripts_new/prepro/")
from cleanandTransform import cleanandTransform

"""
we previously labeled some raw chapter names with chapter categories
eg. {trade facilitation, customs procedures} --> trade facilitation

Now, we setfit our combined tota+ dataset to get chapter categories for all our new raw chapter names
"""

df = pd.read_csv('other_source/totafta.csv')


chapters = pd.unique(df.chapter)

cat = cleanandTransform(filters = [], transformer_package='all-MiniLM-L12-v2')

cat.init_text(chapters)

cat.set_filters(['to_lower', 'remove_url','remove_special_character', 'normalize_unicode', 'remove_punctuation', 'remove_number', 'remove_whitespace'])

cat.process_text()

cat.transform_text()



totalabels = pd.read_csv('other_source/tota_chapters.csv')
labelcat = cleanandTransform(filters = [], transformer_package='all-MiniLM-L12-v2')

labelcat.init_text(totalabels.chapter)

labelcat.set_filters(['to_lower', 'remove_url','remove_special_character', 'normalize_unicode', 'remove_punctuation', 'remove_number', 'remove_whitespace'])

labelcat.process_text()

labelcat.transform_text()

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('trfModels/chaptersFineTuning')

cat.transformer = model
cat.transform_text()
labelcat.transformer = model
labelcat.transform_text()

from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

classifier = LinearSVC(multi_class = 'crammer_singer', max_iter=10000)
classifier = KNeighborsClassifier()

train_embeddings = labelcat.current_embedding

classifier.fit(train_embeddings, totalabels.cluster)
ftaClass = classifier.predict(cat.current_embedding)

# get final output

keyer = pd.concat( [pd.Series(chapters), pd.Series(ftaClass)], axis=1 )
keyer.columns = ['chapter', 'label'] 

df = df.merge(keyer, how='left', left_on='chapter', right_on='chapter')

chapdict=pd.read_csv('core_excels/chapter_dict.csv')
df = df.merge(chapdict, how='left', left_on='label', right_on='chapter_label')

# a bit of manual reclass cuz idk what's going on. trade in goods -> trade in goods

df['chapter_cat'] = df.apply(lambda x: 'trade in goods' if x['chapter'] == 'Trade In Goods' else x['chapter_cat'], axis=1)

df = df.drop(['label', 'chapter_label'], axis=1)

df.to_csv('other_source/tota_chaptersStandardized.csv', index=False)




























from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.cluster import DBSCAN

df = pd.read_csv('other_source/totafta.csv')

chap_names = pd.unique(df.chapter)
#chap_names = df.chapter


# encode chapter names into minilm vectors

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(chap_names)


# eps hyperparameter tuning

from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
nearest_neighbors = NearestNeighbors(n_neighbors=11)
X=embeddings
neighbors = nearest_neighbors.fit(X)
distances, indices = neighbors.kneighbors(X)
distances = np.sort(distances[:,10], axis=0)
fig = plt.figure(figsize=(5, 5))
plt.plot(distances)
plt.xlabel("Points")
plt.ylabel("Distance")
plt.savefig("Distance_curve.png", dpi=300)

# perform DBSCAN clustering 

clustering = DBSCAN(eps=0.7).fit(embeddings)

names = pd.DataFrame()

names['chapter'] = chap_names
names['label'] = clustering.labels_

df['labels'] = clustering.labels_