"""
Our overarching goal is to classify text into those which are 'significant', and those which are not.

A simple example as follows:

//not significant:
"the parties recognise the importance of co-operation in the promotion of competition, economic efficiency, consumer welfare and the curtailment of anti-competitive practices."

//significant:
'to ensure that technical co-operation under this chapter occurs on an ongoing basis, the parties shall designate contact points for technical co-operation and information exchange under this chapter.'


In determining 'significance', I adopt the substantive standard from law. That is, law that creates or defines rights, duties and obligations, and causes of action.

However, for the purposes of this classifier, we only bother with establishing if some right or obligation is constructed, not its degree of enforceability or level of obligation. We leave that task to later.

Data is manually labeled. (Labeler: sam ho)
"""

import pandas as pd
import sys
sys.path.append("scripts/model/")
from cleanandTransform import cleanandTransform

cat = cleanandTransform(filters = [])

df=pd.read_csv('scripts/model/newtota_toSubstantiate.csv')

# Getting train-test data

df=df[df.isSubstantive_label.notnull()]

text = df.text


# Idea: use POS tagging to only select the verbs
import nltk
nltk.download('averaged_perceptron_tagger')
"""
newclause=[]
for clause in text:
    
    
    newlist = [tup[0] for tup in nltk.pos_tag(clause.split()) if len(set(tup).intersection({'MD', })) != 0] 
    # i probably don't want gerunds
    newclause.append(" ".join(newlist))

text=newclause
"""

"""
preprocess and store within the cat object for standardization purposes
"""
cat.set_filters(['to_lower', 'remove_url','remove_special_character', 'normalize_unicode', 'remove_whitespace'])
cat.init_text(text)
cat.process_text()
cat.transform_text()

"""
train test split
"""
from sklearn.model_selection import train_test_split as tts
from sklearn.svm import NuSVC, LinearSVC
from sklearn.preprocessing import StandardScaler

score = []
for i in range(0,100):

    X_train, X_test, y_train, y_test  = tts(cat.current_embedding, df.isSubstantive_label, test_size = 0.2)

    """
    SVM model
    """
    Classifier=LinearSVC(C=1)
    Classifier.fit(X = X_train, y = y_train)

    acscore=Classifier.score(X_test, y_test)
    score.append(acscore)

print(sum(score)/len(score))


"""
keras"""
import keras
from keras import layers
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score

model = keras.Sequential([
        layers.Dense(20, activation="sigmoid"),
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
    epochs=2000,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    #validation_data=(X_test, y_test),
)

y_pred = model.predict(X_test).round()

cm = confusion_matrix(y_true = y_test, y_pred = y_pred)
acscore = accuracy_score(y_true = y_test, y_pred = y_pred)
print(acscore)