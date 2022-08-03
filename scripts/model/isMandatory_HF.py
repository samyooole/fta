from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import pandas as pd

tp = pd.read_csv('core_excels/totaPlus_dp.csv')
df1=tp

df1 = df1[['text', 'isMandatory_label']]
df1=df1.dropna(subset=['isMandatory_label'])
df1['text']=df1.text.astype('str')
df1=df1.rename({'isMandatory_label':'label'},axis=1)


dataset = Dataset.from_pandas(df1)

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized = dataset.map(preprocess_function, batched=True)

dsSub = tokenized.train_test_split() 

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=1)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dsSub["train"],
    eval_dataset=dsSub["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

trainer.evaluate()


from numpy import rint
import numpy as np

yhat = trainer.predict(dsSub['test']).predictions
listlist=[]
for hat in yhat:
    if hat == -0:
        listlist.append([0])
    else:
        listlist.append(list(hat))
yhat=rint(yhat)
y = dsSub['test']['label']

from sklearn.metrics import accuracy_score, confusion_matrix

cm = confusion_matrix(y_true = y, y_pred = yhat)
ac = accuracy_score(y_true = y, y_pred = yhat)



### save!!

trainer.save_model('isMandatory_HF')

######3

tp1 = tp[['text', 'isMandatory_label']]
tp1['text']=tp1.text.astype('str')
tp1=tp1.rename({'isMandatory_label':'label'},axis=1)

dataset_full = Dataset.from_pandas(tp1)

whole_tokenized = dataset_full.map(preprocess_function, batched=True)

classification=trainer.predict(whole_tokenized)

classification = np.round(classification)

classif = np.round(classification.predictions)

tp['isMandatory_predict'] = classif

df1.to_csv('totaPlus_MandatoryPredictNew.csv')




###
df = pd.read_csv("core_excels/totaPlus_dp.csv")
df.text = df['text'].apply(lambda x: str(x))

model = AutoModelForSequenceClassification.from_pretrained("isMandatory_HF", num_labels=1)


trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

df1=df[['text']]

dataset = Dataset.from_pandas(df1)

dstoken = dataset.map(preprocess_function, batched=True)

classification = trainer.predict(dstoken)

"""
IDEA: keep a continuous record of the probability that it is mandatory, use as a continuous variable in subsequent regression
"""

classif_contin = classification.predictions
classif_binary = np.round(classif_contin)

df['isMandatory'] = classif_binary
df['isMandatory_prob'] = classif_contin

df.to_csv('core_excels/totaPlus_MandatoryPredictHF.csv')

