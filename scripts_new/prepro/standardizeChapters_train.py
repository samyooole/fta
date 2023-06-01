import pandas as pd

import sys
sys.path.append("scripts/model/")
from cleanandTransform import cleanandTransform

"""
we previously labeled some raw chapter names with chapter categories
eg. {trade facilitation, customs procedures} --> trade facilitation

Now, we setfit our combined tota+ dataset to get chapter categories for all our new raw chapter names
"""

df = pd.read_csv('other_source/tota_Chapters_training.csv')


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

clusters = pd.unique(totalabels.cluster)

from itertools import combinations
from sentence_transformers import InputExample, losses, evaluation
from itertools import combinations
from torch.utils.data import DataLoader

train_examples=[]

# sameness training
for cluster in clusters:
    indexing = totalabels["cluster"]== cluster
    rawchapters = totalabels.chapter[indexing]

    samecombin = list(combinations(rawchapters,2))

    for combin in samecombin:
        left = combin[0]
        right = combin[1]
        train_example = InputExample(texts=[left, right], label = 0.9)

        train_examples.append(train_example)

    
# differencing training. for simplicity's sake we simply just take one of each and distinguish them from each other
diffchap  = list(combinations(totalabels.drop_duplicates('cluster').chapter,2))
for combin in diffchap:
    left = combin[0]
    right = combin[1]
    train_example = InputExample(texts=[left, right], label = 0.1)
    train_examples.append(train_example)

# train
# load relevant objects
model=cat.transformer
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model=model)
num_epochs=5
model_save_path = 'trfModels/chaptersFineTuning'


"""
may load model from chaptersFineTuning directly next time
"""

model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          output_path=model_save_path)

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


df.to_csv('core_excels/articlesAssigned.csv')



