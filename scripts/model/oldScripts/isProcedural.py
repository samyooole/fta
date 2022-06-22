
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
scaler=StandardScaler()

df=pd.read_csv("text_toSplit.csv")

newdf=df[df['isProcedural_label'].notnull()]

with open('pickles/new_embeddings.pkl', 'rb') as file:
    embed = pickle.load(file) 

# try principal component analysis
from sklearn.decomposition import PCA 

pca=PCA(n_components=200)
embed=pca.fit_transform(embed)

relevant_embed = embed[df['isProcedural_label'].notnull()]


X_train, X_test, y_train, y_test = train_test_split(relevant_embed, newdf['isProcedural_label'], test_size=0.5)



model = keras.Sequential([
        layers.Dense(60, activation="sigmoid"),
        layers.Dense(60, activation="sigmoid"),
        layers.Dense(60, activation="sigmoid"),
        layers.Dense(60, activation="sigmoid"),
        layers.Dense(1, activation="sigmoid", name="predictions")
])

bce = tf.keras.losses.BinaryCrossentropy(reduction='sum')

model.compile(
    optimizer='adam',  # Optimizer
    # Loss function to minimize
    loss=bce,
    # List of metrics to monitor
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

model.fit(
    X_train,
    y_train,
    epochs=5000,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    #validation_data=(X_test, y_test),
)

y_pred = model.predict(X_test).round()

cm = confusion_matrix(y_true = y_test, y_pred = y_pred)
acscore = accuracy_score(y_true = y_test, y_pred = y_pred)
print(acscore)


from sklearn import svm
clf = svm.SVR()
acscorelist=[]
for i in range(0,5):

    X_train, X_test, y_train, y_test = train_test_split(relevant_embed, newdf['isProcedural_label'], test_size=0.2, random_state=i)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    y_pred = [round(elem) for elem in y_pred]

    

    cm = confusion_matrix(y_true = y_test, y_pred = y_pred)
    acscore = accuracy_score(y_true = y_test, y_pred = y_pred)

    acscorelist.append(acscore)

sum(acscorelist)/len(acscorelist)


clf.fit(relevant_embed, newdf['isProcedural_label'])
y_pred = clf.predict(embed)
y_pred = [round(elem) for elem in y_pred]

df['isProcedural_prediction'] = y_pred

df.to_csv('text_toCategorize.csv')

############


X_train, X_test, y_train, y_test = train_test_split(relevant_embed, newdf['isProcedural_label'], test_size=0.5)

shape=embed.shape[1]

model = keras.Sequential([
        layers.LSTM(64, input_shape = (shape,1), activation='relu', return_sequences=True),
        layers.BatchNormalization(),
        layers.Dense(1, activation="sigmoid", name="predictions")
])


model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="sgd",
    metrics=["accuracy"],
)	

model.fit(
    X_train,
    y_train,
    epochs=1,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    #validation_data=(X_test, y_test),
)

y_pred = model.predict(X_test).round()

cm = confusion_matrix(y_true = y_test, y_pred = y_pred)
acscore = accuracy_score(y_true = y_test, y_pred = y_pred)
print(acscore)