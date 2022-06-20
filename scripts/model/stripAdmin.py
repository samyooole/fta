
from sentence_transformers import SentenceTransformer
import datasets


dsdf = datasets.Dataset.from_csv('text_toStripAdmin.csv')
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

dsdf = datasets.Dataset.from_csv('text_toStripAdmin.csv')
new_embeddings = model.encode(dsdf['text'], show_progress_bar=True)

classification = svmodel.predict(new_embeddings)

import pandas as pd
import pickle
df=pd.read_csv('text_toStripAdmin.csv')

df['predictions'] = classification

df.to_csv('text_toSplit.csv')

with open('pickles/new_embeddings.pkl', 'wb') as file:
    pickle.dump(new_embeddings, file)

## now, classify all text


#################################################################################################################################
"""
never mind lol turns out svm works the best :)"""

""""
try deep learning. mnist
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

input_shape = X_train.shape[1]

model = keras.Sequential(
    [   
        layers.Bidirectional(layers.LSTM(input_shape/2, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(input_shape/2)),
        layers.Dense(1, activation="sigmoid")
    ]
)

model.compile(loss='mse')
# This builds the model for the first time:
model.fit(X_train, np.array([y_train]).T, batch_size=32, epochs=2000)

model.evaluate(X_test, np.array([y_test]).T)

del model
keras.backend.clear_session()


"""
lstm
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

regressor.add(LSTM(units = 5, return_sequences = True, input_shape = (X_train.shape[1], 1)))

regressor.add(LSTM(units = 5, return_sequences = True))

regressor.add(LSTM(units = 5, return_sequences = True))

regressor.add(LSTM(units = 5))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, np.array([y_train]).T, epochs = 100, batch_size = 32)
