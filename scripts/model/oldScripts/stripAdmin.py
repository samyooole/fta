
from sentence_transformers import SentenceTransformer
import datasets


dsdf = datasets.Dataset.from_csv('text_prelabel.csv')
dsdf = dsdf.filter(lambda x: x['label'] != None)

model = SentenceTransformer('all-mpnet-base-v2')

embeddings = model.encode(dsdf['text'])

from sklearn.model_selection import train_test_split


from sklearn import svm
clf = svm.SVR()


X_train, X_test, y_train, y_test = train_test_split(embeddings, dsdf['label'], test_size=0.2, random_state=1999)
svmodel = clf.fit(X_train, y_train)

y_pred = svmodel.predict(X_test)

y_pred = [round(elem) for elem in y_pred]

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_true = y_test, y_pred = y_pred)
acscore = accuracy_score(y_true = y_test, y_pred = y_pred)

dsdf = datasets.Dataset.from_csv('text_prelabel.csv')
new_embeddings = model.encode(dsdf['text'], show_progress_bar=True)

svmodel=clf.fit(new_embeddings, dsdf['label'])
classification = svmodel.predict(new_embeddings)

import pandas as pd
import pickle
df=pd.read_csv('text_prelabel.csv')

df['predictions'] = classification

df.to_csv('text_toSplit.csv')

with open('pickles/new_embeddings.pkl', 'wb') as file:
    pickle.dump(new_embeddings, file)


"""
keep only the non-admin text stuff
"""
newdf = df[df['predictions'] < 0.5]
relevant_embed = new_embeddings[df['predictions'] < 0.5]

newdf.to_csv('pickles/text.csv')
with open('pickles/relevant_embeddings.pkl', 'wb') as file:
    pickle.dump(relevant_embed, file)