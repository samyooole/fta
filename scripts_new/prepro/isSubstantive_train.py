"""
Our overarching goal is to classify text into those which are 'significant', and those which are not.

A simple example as follows:

//not significant:
"the parties recognise the importance of co-operation in the promotion of competition, economic efficiency, consumer welfare and the curtailment of anti-competitive practices."

//significant:
'to ensure that technical co-operation under this chapter occurs on an ongoing basis, the parties shall designate contact points for technical co-operation and information exchange under this chapter.'


In determining 'significance', I adopt the substantive standard from law. That is, law that creates or defines rights, duties and obligations, and causes of action.

However, for the purposes of this classifier, we only bother with establishing if some right or obligation is constructed, not its degree of enforceability or level of obligation. We leave that task to later.

Data is manually labeled. (Labeler: sam ho)
"""



import os
import pandas as pd


df=pd.read_csv('other_source/isSubstantive_traindata.csv')

# Getting train-test data

df=df[df.isSubstantive_label.notnull()].reset_index()

df['isSubstantive_label'] = df.isSubstantive_label.astype(int)


# following follows from text classification guide by HF https://huggingface.co/docs/transformers/tasks/sequence_classification

from datasets import Dataset


ds = Dataset.from_pandas(df[['text', 'isSubstantive_label']].rename({'isSubstantive_label': 'labels'},axis=1))

ds = ds.train_test_split(test_size = 0.2)


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized = ds.map(preprocess_function, batched=True)

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

import evaluate

accuracy = evaluate.load("accuracy")

import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


id2label = {0: "isNotSubstantive", 1: "isSubstantive"}
label2id = {"isNotSubstantive": 0, "isSubstantive": 1}

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir="trfModels/isSubstantive",
    learning_rate=2e-5,
    per_device_train_batch_size=5,
    per_device_eval_batch_size=20,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

import torch
torch.cuda.empty_cache()

trainer.train()


trainer.save_model('trfModels/isSubstantive')