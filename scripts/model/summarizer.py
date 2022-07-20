from transformers import pipeline
import pandas as pd

df = pd.read_csv('core_excels/totaPlus_bypass.csv')
df = df[df['isSubstantive_predict'] == 1]
df=df.reset_index(drop=True)

df['text'] = df.text.apply(lambda x: x.replace("-", "").strip())


# split by periods, then explode. this is because clauses are sometimes composed of multiple sentences, which generally correspond to multiple core actions

df['text'] = df.text.apply(lambda x: x.split("."))
df['text'] = df.text.apply(lambda x: [item for item in x if item != ''])

df = df.explode('text')
df=df.drop('Unnamed: 0', axis=1)
df=df.reset_index(drop=True)
df['text'] = df.text.apply(lambda x: x.strip())


classifier = pipeline("summarization")

classifier(df.text[0])