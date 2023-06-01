"""
- takes in annotated text jsons from label studio
- trains upon the spacy ner framework

some reminders for label studio:
run on cli
    label-studio start

some reminders on spacy v3.0 training:
- use gpu?
run on cli (all these run on global spacy, not venv spacy)
<to download base model> python -m spacy download en_core_web_lg </>
<download config file from spacy website>
<fill config file with write paths for train and dev>
<autofill config file> python -m spacy init fill-config base_config.cfg config.cfg </>
python -m spacy train config.cfg --output ./output --gpu-id 0
    

"""

import json
import pandas as pd
import spacy
from spacy.tokens import DocBin
import json
from tqdm import tqdm


with open('scripts/model/rob/rob_labels.json') as f:
    data=json.load(f)

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
try model on new words
"""
import pandas as pd

tp = pd.read_csv('core_excels/totaPlus_MandatoryPredict.csv')

newtext = tp.text.astype('str')


#################################
"""
first run in command line
python -m spacy download en_core_web_lg
"""

# define our training data to TRAIN_DATA
TRAIN_DATA = gold_format

# create a blank model
nlp = spacy.blank('en')

def create_training(TRAIN_DATA):
    db = DocBin()
    for text, annot in tqdm(TRAIN_DATA):
        doc = nlp.make_doc(text)
        ents = []

        # create span objects
        for start, end, label in annot["entities"]:
            span = doc.char_span(start, end, label=label, alignment_mode="contract") 

            # skip if the character indices do not map to a valid span
            if span is None:
                print("Skipping entity.")
            else:
                ents.append(span)
                # handle erroneous entity annotations by removing them
                try:
                    doc.ents = ents
                except:
                    # print("BAD SPAN:", span, "\n")
                    ents.pop()
        doc.ents = ents

        # pack Doc objects into DocBin
        db.add(doc)
    return db

TRAIN_DATA_DOC = create_training(TRAIN_DATA)

# Export results (here I add it to a TRAIN_DATA folder within the directory)
TRAIN_DATA_DOC.to_disk("./TRAIN_DATA/TRAIN_DATA.spacy")







########
"""
try
"""

nlp1 = spacy.load("./output.gpu/model-best")
example = newtext[40006]
doc = nlp1(example)
doc

for ent in doc.ents:
    print(ent, ent.label_)
