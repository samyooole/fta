import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, np_utils
import pickle
import numpy as np

df=pd.read_csv("text_toCategorize.csv")

import datasets
dsdf = datasets.Dataset.from_csv('text_toCategorize.csv')
dsdf = dsdf.filter(lambda x: x['catlabel'] != None)

Y = dsdf['catlabel']

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

labels=dummy_y.shape[1]


##################
import keras
from keras import layers
import pickle

with open('pickles/new_embeddings.pkl', 'rb') as file:
    embed = pickle.load(file) 


from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-mpnet-base-v2')
relevant_embed = model.encode(dsdf['text'], show_progress_bar=True)

#tf.reshape(relevant_embed, [None, relevant_embed.shape[0], relevant_embed.shape[1]])

X_train,X_test,Y_train,Y_test = train_test_split(relevant_embed,dummy_y,test_size=0.2)

model = keras.Sequential([
        layers.Dense(100, activation="sigmoid"),
        layers.Dropout(0.1),
        layers.Dense(dummy_y.shape[1], activation="sigmoid", name="predictions")
])

model.compile(loss='categorical_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

model.fit(X_train,Y_train,epochs=400)

accr = model.evaluate(X_test,Y_test)

ytensor = model.predict(X_test)

print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

catlist=["preferential tariffs",
"services",
"investments",
"movement of persons",
"intellectual property rights",
"public procurement",

"competition policy",

"technical barriers to trade",
"sanitary and phytosanitary measures",
"trade facilitation",
"trade remedies",
"e-commerce",

"environment",
"labour market",
"export restrictions",
"rules of origin"
]

model.fit(relevant_embed,dummy_y,epochs=400)

zhat = model.predict(embed)
cathat = [catlist[np.argmax(zhat_i)] for zhat_i in zhat]

pd.Series(cathat).to_csv('withnewTrainingData.csv')


df['cathat'] = cathat

df.to_csv("afterCategory.csv")





















############### try svc
from sklearn.svm import LinearSVC, NuSVC, SVC



model = SVC()

X_train,X_test,Y_train,Y_test = train_test_split(relevant_embed,dsdf['catlabel'],test_size=0.4)

model.fit(X_train, Y_train)
yhat=model.predict(X_test)

sum(yhat==Y_test)/len(yhat)


model.fit(relevant_embed, dsdf['catlabel'])
zhat=model.predict(embed)

df['cat_prediction'] = zhat

df.to_csv("svm.csv")

####### dump

X_train,X_test,Y_train,Y_test = train_test_split(text,dummy_y,test_size=0.4)

max_words = 1000
max_len = 300
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = pad_sequences(sequences,maxlen=max_len)

def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,500,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(64,name='FC1')(layer)
    layer = Activation('sigmoid')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(labels,name='out_layer')(layer)
    layer = Activation('softmax')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

model = RNN()
model.summary()
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(sequences_matrix,Y_train,epochs=30)

test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = pad_sequences(test_sequences,maxlen=max_len)

accr = model.evaluate(test_sequences_matrix,Y_test)

print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))