from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split as tts
import pandas as pd

df = pd.read_csv("scripts/model/newtota_toSubstantiate.csv")
df.text = df['text'].apply(lambda x: str(x))


import sys
sys.path.append("scripts/model/")
from cleanandTransform import cleanandTransform

cat = cleanandTransform(filters = [], transformer_package='all-MiniLM-L12-v2')

cat.set_filters(['to_lower', 'remove_url','remove_special_character', 'normalize_unicode', 'remove_whitespace'])
cat.init_text(df.text)
cat.process_text()

df.text = cat.current_text # try with uncased and appropriate filtering

df['label']=df.apply(lambda x: x['isSubstantive_label'] if pd.isnull(x['isSubstantive_cordon_label']) else x['isSubstantive_cordon_label'], axis=1)
df1=df.drop(['chapter', 'article', 'clauseno', 'fta_name', 'rta_id', 'toa', 'parties', 'dif', 'isSubstantive_label', 'isSubstantive_cordon_label'], axis=1)

df1=df1.dropna()

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
    num_train_epochs=20,
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

# 5 epochs: 83% test accuracy
# (20+5) epochs: 88% test accuracy
# 100 epochs: also around 88% test accuracy. petering out

# now, we attempt for around 20 epochs

with open('working_pickles/hfmodel_100epochs_standard.pkl', 'wb') as f:
    pickle.dump(trainer, f)



########################################### take our previous hfmodel ~ 88% test accuracy


df = pd.read_csv("scripts/model/newtota_toSubstantiate.csv")
df.text = df['text'].apply(lambda x: str(x))

model = AutoModelForSequenceClassification.from_pretrained("hfmodel_cased", num_labels=1)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dsSub["train"],
    eval_dataset=dsSub["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

df1=df.drop(['chapter', 'article', 'clauseno', 'fta_name', 'rta_id', 'toa', 'parties', 'dif', 'isSubstantive_label', 'isSubstantive_cordon_label'], axis=1)

df1=df1.dropna()

dataset = Dataset.from_pandas(df1)

dstoken = dataset.map(preprocess_function, batched=True)

classification = trainer.predict(dstoken)

classification = np.round(classification)

classif = np.round(classification.predictions)

df1['isSubstantive_predict'] = classif

df1.to_csv('scripts/model/newtota_SubstantiatePredict_HF.csv')