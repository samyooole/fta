"""
we conduct a preliminary NER analysis of PTA texts.

some brief goals, in order ot increasing complexity:
- identify explicity named entities: eg. ASEAN, Brunei, Cambodia (as a baseline spacy will do pretty well on this)
- identify implicitly named entities: eg. the Parties, the Agreement, the arbitral panel
"""

import spacy
import pandas as pd
import random
import re

tp = pd.read_csv('core_excels/totaPlus_bypass.csv')
tp['text']=tp.text.astype('str')
nlp = spacy.load('en_core_web_trf')

"""
Model #1: 
- to identify all named entities, regardless of whether they are explicitly or implicitly named
- for example, "the Parties" should be picked up on as much as "ASEAN", "New Zealand", "Japan", etc.
- to do this, we work off of spacy's pre-trained models
- then, further train based on manual labelling
"""

tp = tp.sample(1000)

tp['ner'] = tp['text'].apply(lambda x: nlp(x).ents)


tp.to_csv('checkner2.csv')


"""
reload back to continue training spacy model
"""

furthertrain = pd.read_csv('checkner2.csv')
furthertrain = furthertrain[furthertrain.newtag.notnull()]
furthertrain=furthertrain.reset_index(drop=True)
furthertrain['newtag'] = furthertrain['newtag'].apply(lambda x: x.split(", ") if "," in x else x)

furthertrain = furthertrain.explode('newtag').reset_index(drop=True)

furthertrain['left_right'] = furthertrain.apply(lambda x: re.search(pattern = x['newtag'], string = x['text']).span() if re.search(pattern = x['newtag'], string = x['text']) is not None else None, axis=1)

furthertrain = furthertrain[furthertrain.left_right.notnull()]
furthertrain=furthertrain.reset_index(drop=True)

train_data = []

for row in furthertrain.iterrows():
    text = row[1].text
    left_right = row[1].left_right
    updatee=(text, {"entities": [(left_right[0], left_right[1], "ORG")]})

    train_data.append(updatee)


"""
get spacy ner model to continue training
"""
ner = nlp.get_pipe("ner")

for _, annotations in train_data:
  for ent in annotations.get("entities"):
    ner.add_label(ent[2])


# Import requirements
from spacy.training import Example
    

optimizer = nlp.resume_training()
for itn in range(100): # repeat and shuffle
    random.shuffle(train_data)
    for raw_text, entity_offsets in train_data:
        doc = nlp.make_doc(raw_text)
        example = Example.from_dict(doc, entity_offsets)
        nlp.update([example], sgd=optimizer)
nlp.to_disk("/output")


from spacy.cli.train import train

train("config.cfg", overrides={"paths.train": "./train.spacy", "paths.dev": "./dev.spacy"})

furthertrain['ner_new'] = furthertrain['text'].apply(lambda x: nlp(x).ents)


# some shebangs
def trim_entity_spans(data: list) -> list:
    """Removes leading and trailing white spaces from entity spans.

    Args:
        data (list): The data to be cleaned in spaCy JSON format.

    Returns:
        list: The cleaned data.
    """
    invalid_span_tokens = re.compile(r'\s')

    cleaned_data = []
    for text, annotations in data:
        entities = annotations['entities']
        valid_entities = []
        for start, end, label in entities:
            valid_start = start
            valid_end = end
            while valid_start < len(text) and invalid_span_tokens.match(
                    text[valid_start]):
                valid_start += 1
            while valid_end > 1 and invalid_span_tokens.match(
                    text[valid_end - 1]):
                valid_end -= 1
            valid_entities.append([valid_start, valid_end, label])
        cleaned_data.append([text, {'entities': valid_entities}])

    return cleaned_data

train_data = trim_entity_spans(train_data)







"""

try the config way

"""

import pandas as pd
from tqdm import tqdm
import spacy
from spacy.tokens import DocBin

nlp = spacy.blank("en") # load a new spacy model
db = DocBin() # create a DocBin object

for text, annot in tqdm(train_data): # data in previous format
    doc = nlp.make_doc(text) # create doc object from text
    ents = []
    for start, end, label in annot["entities"]: # add character indexes
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    doc.ents = ents # label the text with the ents
    db.add(doc)

db.to_disk("./train.spacy") # save the docbin object