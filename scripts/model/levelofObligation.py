"""
we presently work with two levels of obligation, operating off of the core verb phrase:

- optional (Japan may give Singapore $2)
- mandatory (Japan shall give Singapore $2)
[note that we are only getting the mandatory nature or not. we do not yet consider the 

- we are also presently operating under the assumption that all are obligations. few if any rights are established under FTAs

"""


import os

from sklearn.linear_model import LogisticRegression
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7/bin")
import pandas as pd
import sys
sys.path.append("scripts/model/")
from cleanandTransform import cleanandTransform

cat = cleanandTransform(filters = [], transformer_package='all-MiniLM-L12-v2')

df=pd.read_csv('scripts/model/totaPlus_preMandate.csv')

# Getting train-test data

df=df[df.isMandatory_label.notnull()].reset_index()

#try using cvp instead of entire text?
text = df.cvp

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


# train test split HERE

from sklearn.model_selection import train_test_split as tts

X_train, X_test, y_train, y_test  = tts(cat.current_embedding, df.isMandatory_label, test_size = 0.2)

# load and add combinations of 1s

list_of_ones = text[y_train[y_train == 1].index]
combination_of_ones = list(combinations(list_of_ones, r=2))
for combination in combination_of_ones:
    left = combination[0]
    right = combination[1]
    train_example = InputExample(texts=[left, right], label = 0.9)

    train_examples.append(train_example)

# load and add combinations of 0s
list_of_zeroes = text[y_train[y_train == 0].index]
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
model_save_path = 'scripts/model/MandatoryFineTuning'


"""
may load model from MandatoryFineTuning directly next time
"""

model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          output_path=model_save_path)

cat.transformer = model
cat.transform_text()

nuX_test = cat.current_embedding[y_test.index]
nuX_train = cat.current_embedding[y_train.index]

# now, push into another standard classifier. does not really matter which

from sklearn.svm import NuSVC, LinearSVC

Classifier=NuSVC()
Classifier.fit(X = nuX_train, y = y_train)

acscore=Classifier.score(nuX_test, y_test)

# we get a 96.7% test accuracy

######################################################################
"""
start here if simply loading model
"""


import os

from sklearn.linear_model import LogisticRegression
import pandas as pd
import sys
sys.path.append("scripts/model/")
from cleanandTransform import cleanandTransform
from sentence_transformers import SentenceTransformer

df=pd.read_csv('core_excels/totaPlus_dp.csv')
text=df.cvp.astype('str')
newcat = cleanandTransform(filters = [], transformer_package='all-MiniLM-L12-v2')
newcat.model = SentenceTransformer('trfModels/MandatoryFineTuning')

newcat.set_filters(['to_lower', 'remove_url','remove_special_character', 'normalize_unicode', 'remove_whitespace'])
newcat.init_text(text)
newcat.process_text()
newcat.transform_text()

from sklearn.svm import NuSVC

indexer = df.isMandatory_label.notnull()
X_train = newcat.current_embedding[indexer]
y_train = df[df.isMandatory_label.notnull()].isMandatory_label

Classifier=NuSVC()
Classifier.fit(X = X_train, y = y_train)

array = Classifier.predict(newcat.current_embedding)

df['isMandatory_predict'] = array

df.to_csv("scripts/model/totaPlus_MandatoryPredict.csv")