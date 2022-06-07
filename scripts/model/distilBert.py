
from transformers import AutoTokenizer
import datasets

dsdf = datasets.Dataset.from_csv('text_postlabel.csv')
dsdf = dsdf.filter(lambda x: x['ntb'] != None)
dsdf = dsdf.rename_column("ntb", "label")

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True)


tokenized_ntb = dsdf.map(preprocess_function, batched=True)

dsntb = tokenized_ntb.train_test_split()

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

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
    train_dataset=dsntb["train"],
    eval_dataset=dsntb["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import pandas as pd

yhat = trainer.predict(dsntb['test'])[0]
yhat = np.round(yhat)
yhat = np.ndarray.flatten(yhat)
y = dsntb['test']['label']

cm = confusion_matrix(y_true = y, y_pred = yhat)
acscore = accuracy_score(y_true = y, y_pred = yhat)

validate=datasets.Dataset.from_csv('validate.csv')
validate = validate.map(preprocess_function, batched=True)
zhat = trainer.predict(validate)[0]

validate = pd.read_csv('validate.csv')
validate['zhat'] = zhat

validate.to_csv('validate_d.csv')
