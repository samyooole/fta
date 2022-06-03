from transformers import RobertaTokenizer, RobertaModel
import pickle
import pandas as pd

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Import our corpus

file = open('pickles/fullytreated_corpus.pkl', 'rb')
text = pickle.load(file) 

text = [para for para in text if len(str.split(para)) <= 400]

textcol = pd.Series(text)
textcol.to_csv('text_prelabel.csv')

token_text = [tokenizer(para, return_tensors = 'pt') for para in text] # first brush CHANGE
model = RobertaModel.from_pretrained("roberta-base")

roberta_matrix = []
for id, item in enumerate(token_text):

    outputs = model(**item) # idk if this is what the output of the robertamodel gives
    poolvector = outputs[1] # gives pooled vector
    roberta_matrix.append(poolvector)

    print(id/len(token_text))
