"""
- takes in annotated text jsons from label studio
- trains upon the space ner framework

"""

import json
import sys
sys.path.append("scripts/model/rob")
from NERModel import NERModel

with open('scripts/model/rob/rob_labels.json') as f:
    data=json.load(f)

data[0]['annotations']

def LS_to_spacy(jsondata):
    """
    """
    outputlist = []
    for line in jsondata:
        

        text = line['data']['text'].strip('"')
        gold_list = [ (annot['value']['start'], annot['value']['end'], annot['value']['labels'][0]) for annot in line['annotations'][0]['result']]

        gold_dict = {"entities": gold_list}
    
        outputlist.append((text, gold_dict))
    return outputlist
    
gold_format = LS_to_spacy(data)

"""
coerce to nermodel class
"""

nm = NERModel()
nm.fit(gold_format)

"""
try model on new words
"""
import pandas as pd

tp = pd.read_csv('core_excels/totaPlus_MandatoryPredict.csv')

newtext = tp.text.astype('str')


example = newtext[0]
nm.ner_model(example).ents
doc = nlp(example)

doc.ents