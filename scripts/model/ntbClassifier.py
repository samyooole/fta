from keras.models import Sequential
from keras.layers import Dense 
import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
import pickle
import pandas as pd
import numpy as np

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')


df = pd.read_csv('text_postlabel.csv')
df = df.dropna()

text=df.text
y = df.ntb



token_text = [tokenizer(para, return_tensors = 'pt') for para in text] # first brush CHANGE
model = RobertaModel.from_pretrained("roberta-base")

roberta_matrix = []
for id, item in enumerate(token_text):

    outputs = model(**item) # idk if this is what the output of the robertamodel gives
    poolvector = outputs[1][0] # gives pooled vector
    roberta_matrix.append(poolvector)

    print(id/len(token_text))

roberta_matrix = roberta_matrix

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(roberta_matrix, y, epochs=150, batch_size=10)


