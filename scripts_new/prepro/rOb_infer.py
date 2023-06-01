

import spacy

nlp1 = spacy.load("./trfModels/spacy_output/model-best")

import pandas as pd

df=  pd.read_csv('other_source/tota_Substantiated.csv')


from tqdm import tqdm

obl = [nlp1(item).ents for item in tqdm(df.text.to_list())]

obl_text = [[subitem.text for subitem in item] for item in tqdm(obl)]


df['obl'] = obl_text


# explode obligations downwards + remove empty obligations

df = df.explode('obl')

df = df.dropna(subset=['obl'])

df.to_csv('other_source/tota_Obligated.csv')