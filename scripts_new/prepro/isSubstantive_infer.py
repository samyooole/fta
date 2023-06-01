

from transformers import pipeline

classifier = pipeline("text-classification", model="trfModels/isSubstantive")


import os
import pandas as pd


df=pd.read_csv('other_source/tota_chaptersStandardized.csv')

df = df.dropna(subset='text')

df = df.reset_index(False)

#batch inference for faster inference

from transformers import pipeline
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from datasets import Dataset

ds = Dataset.from_pandas(df[['text']])



from datasets import Dataset
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from optimum.bettertransformer import BetterTransformer


device = torch.device('cuda')

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

model_ft = AutoModelForSequenceClassification.from_pretrained('trfModels/isSubstantive').to(device)

model_ft = BetterTransformer.transform(model_ft, keep_original_model=True)


classifier = pipeline("text-classification", model=model_ft, tokenizer = tokenizer, device=0 )

###
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets

lol = []
for out in tqdm(classifier(KeyDataset(ds, 'text'), batch_size=256, truncation='only_first'), total = len(ds)):
    lol.append(out)

lol = pd.DataFrame(lol)

df['isSubstantive_predict'] = lol['label']

df = df.drop('index', axis=1)

df.to_csv('other_source/tota_Substantiated.csv', index=False)
