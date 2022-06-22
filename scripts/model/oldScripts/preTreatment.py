import pandas as pd
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation
import spacy
import pickle

nlp = spacy.load('en_core_web_sm')

def treatText(dftext):
    newtext=[remove_stopwords(text) for text in dftext] # remove stopwords
    newtext=[strip_punctuation(text) for text in newtext] # remove punctuation

    newtext1=[]
    for idx,text in enumerate(newtext):
        words=text.split()
        newords = [word for word in words if len(word) > 1]
        newords = [word for word in newords if word.isdigit() is False]
        #newords = [word.lemma_ for word in nlp(" ".join(newords))]
        newtext1.append(newords)
        print(idx/len(newtext))

    treated_text = [" ".join(words) for words in newtext1]
    return treated_text

# create embeddings for text

from sentence_transformers import SentenceTransformer
df=pd.read_csv("text_toCategorize.csv")
newtext=treatText(df['text'])

model = SentenceTransformer('all-mpnet-base-v2')
new_embeddings = model.encode(newtext, show_progress_bar=True)

with open('pickles/new_embeddings.pkl', 'wb') as file:
    pickle.dump(new_embeddings, file)

