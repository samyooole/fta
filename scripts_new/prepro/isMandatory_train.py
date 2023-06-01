from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import pandas as pd

tp = pd.read_csv('other_source/isMandatory_train.csv')
df1=tp

df1 = df1[['text', 'isMandatory_label']]
df1=df1.dropna(subset=['isMandatory_label'])
df1['isMandatory_label'] = df1.isMandatory_label.astype(int)
df1['text']=df1.text.astype('str')
df1=df1.rename({'isMandatory_label':'label'},axis=1)


dataset = Dataset.from_pandas(df1)

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized = dataset.map(preprocess_function, batched=True)

dsSub = tokenized.train_test_split() 

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

id2label = {0: "isNotMandatory", 1: "isMandatory"}
label2id = {"isNotMandatory": 0, "isMandatory": 1}

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=5,
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

trainer.save_model('trfModels/isMandatory')

