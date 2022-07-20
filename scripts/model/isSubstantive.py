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

import os

from sklearn.linear_model import LogisticRegression
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7/bin")
import pandas as pd
import sys
sys.path.append("scripts/model/")
from cleanandTransform import cleanandTransform

cat = cleanandTransform(filters = [], transformer_package='all-MiniLM-L12-v2')

df=pd.read_csv('scripts/model/newtota_toSubstantiate.csv')

# Getting train-test data

df=df[df.isSubstantive_label.notnull()].reset_index()

text = df.text

"""
preprocess and store within the cat object for standardization purposes
"""
cat.set_filters(['to_lower', 'remove_url','remove_special_character', 'normalize_unicode', 'remove_whitespace'])
cat.init_text(text)
cat.process_text()
cat.transform_text()

"""
continue training the sentence transformer model. (TO WRITE INTO CAT CLASS)
- we take all the 1s and set combinations of them to have a cosine sim of 0.9
- similarly for the 0s
- then, for the combinations of strictly 1,0s, we assign them a cosine sim of 0.1
- then, continue training the sentence transformer model within the cat object
"""

from sentence_transformers import InputExample, losses, evaluation
from itertools import combinations
from torch.utils.data import DataLoader
from random import sample

train_examples = []
text = pd.Series(cat.current_text)

# load and add combinations of 1s
list_of_ones = text[df['isSubstantive_label'] == 1]
combination_of_ones = list(combinations(list_of_ones, r=2))
for combination in combination_of_ones:
    left = combination[0]
    right = combination[1]
    train_example = InputExample(texts=[left, right], label = 0.9)

    train_examples.append(train_example)

# load and add combinations of 0s
list_of_zeroes = text[df['isSubstantive_label'] == 0]
combination_of_zeroes = list(combinations(list_of_zeroes, r=2))
for combination in combination_of_zeroes:
    left = combination[0]
    right = combination[1]
    train_example = InputExample(texts=[left, right], label = 0.9)

    train_examples.append(train_example)

# load and add combinations of {1,0}s
for one in list_of_ones:
    for zero in list_of_zeroes:
        train_example = InputExample(texts=(one, zero), label = 0.1)
        train_examples.append(train_example)

# load relevant objects
model=cat.transformer
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model=model)
num_epochs=5
model_save_path = 'scripts/model/substantiveFineTuning'


"""
may load model from substantiveFineTuning directly next time
"""

model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          output_path=model_save_path)

cat.transformer = model
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
    Classifier=NuSVC()
    Classifier.fit(X = X_train, y = y_train)

    acscore=Classifier.score(X_test, y_test)
    score.append(acscore)

print(sum(score)/len(score))

with open('working_pickles/trmodel_30000sample_additivemodel.pkl', 'wb') as f:
    pickle.dump(model,f)




df=pd.read_csv('scripts/model/newtota_toSubstantiate.csv')

df=df[df.isSubstantive_label.notnull()].reset_index()
Classifier = GaussianNB()
Classifier.fit(X=cat.current_embedding, y=df.isSubstantive_label)






######################### testing

df=pd.read_csv('scripts/model/newtota_toSubstantiate.csv')
df=df[df.isSubstantive_cordon_label.notnull()].reset_index()
text=df.text
newcat = cleanandTransform(filters = [], transformer_package='all-MiniLM-L12-v2')
newcat.set_filters(['to_lower', 'remove_url','remove_special_character', 'normalize_unicode', 'remove_whitespace'])
newcat.init_text(text)
newcat.process_text()
newcat.transformer = cat.transformer
newcat.transform_text()

Classifier.score(newcat.current_embedding, df.isSubstantive_cordon_label)


############################ output text for sanity check
df=pd.read_csv('scripts/model/newtota_toSubstantiate.csv')
text=df.text
newcat.set_filters(['to_lower', 'remove_url','remove_special_character', 'normalize_unicode', 'remove_whitespace'])
newcat.init_text(text)
newcat.process_text()
newcat.transformer = cat.transformer
newcat.transform_text()
array = Classifier.predict(newcat.current_embedding)

df['isSubstantive_predict'] = array

df.to_csv("scripts/model/newtota_SubstantiatePredict.csv")



##############################
"""
for fresh running through, start code here
"""

import os

from sklearn.linear_model import LogisticRegression
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7/bin")
import pandas as pd
import sys
sys.path.append("scripts/model/")
from cleanandTransform import cleanandTransform
from sentence_transformers import SentenceTransformer

df=pd.read_csv('scripts/model/newtota_toSubstantiate.csv')
text=df.text
cat = cleanandTransform(filters = [], transformer_package='all-MiniLM-L12-v2')
cat.model = SentenceTransformer('trfModels/substantiveFineTuning')

cat.set_filters(['to_lower', 'remove_url','remove_special_character', 'normalize_unicode', 'remove_whitespace'])
cat.init_text(text)
cat.process_text()
cat.transform_text()