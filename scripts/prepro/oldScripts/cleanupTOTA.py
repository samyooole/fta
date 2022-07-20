import pandas as pd
import sys
sys.path.append("scripts/model/")
from cleanandTransform import cleanandTransform

# load TOTA output from parseTOTA.py

tota = pd.read_csv("scripts/prepro/totafta.csv")

# do explicit cleaning first

text=tota.text.astype('str')

cat = cleanandTransform(filters = [], transformer_package='all-MiniLM-L12-v2')

cat.set_filters(['remove_url','remove_special_character', 'normalize_unicode', 'remove_whitespace'])
cat.init_text(text)
cat.process_text()



tota.text = cat.current_text

tota.to_csv('scripts/prepro/totafta.csv')



















































# now train the garbage detector

garbdf = pd.read_csv('core_excels/train_isGarbage.csv')
garbdf['label']

garbdf = garbdf[pd.notnull(garbdf['label'])][['text', 'label']]
garbtext=garbdf.text

garbcat = cleanandTransform(filters = [], transformer_package='all-MiniLM-L12-v2')

garbcat.set_filters(['remove_url','remove_special_character', 'normalize_unicode', 'remove_whitespace'])
garbcat.init_text(garbtext)
garbcat.process_text()
garbcat.transform_text()

"""
setfitting
"""
from itertools import combinations
from sentence_transformers import InputExample, losses
from itertools import combinations
from torch.utils.data import DataLoader
train_examples=[]
# load and add combinations of 1s

list_of_ones = garbdf[garbdf['label'] == 1].text
combination_of_ones = list(combinations(list_of_ones, r=2))
for combination in combination_of_ones:
    left = combination[0]
    right = combination[1]
    train_example = InputExample(texts=[left, right], label = 0.9)

    train_examples.append(train_example)

# load and add combinations of 0s
list_of_zeroes = garbdf[garbdf['label'] == 0].text
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
model=garbcat.transformer
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model=model)
num_epochs=5
model_save_path = 'trfModels/GarbageFineTuning'

"""
may load model from MandatoryFineTuning directly next time
"""

model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          output_path=model_save_path)









