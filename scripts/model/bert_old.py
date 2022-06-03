import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import pickle
import sys
sys.path.append("scripts/prepro/")
from corpusManagement import getcorpusbyParas, getKMiddle

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Import our corpus

file = open('pickles/fullytreated_corpus.pkl', 'rb')
text = pickle.load(file) 
paralist=[]
for para in text:
    
    modified_para = "[CLS] " + para + " [SEP]" # important step here, we have to configure special tags so we can use BERT

    paralist.append(modified_para)

# Before tokenizing, check each para if it is longer than 512 (max for bert) if so, subset the middle 512 tokens
for id, para in enumerate(paralist):
    if len(str.split(para)) > 512:
        tokens = str.split(para)
        middletokens = getKMiddle(tokens, K=512)
        paralist[id] = " ".join(middletokens)


# not sure what's going on, so let's just drop superlong stuff first
newparalist=[]
for id, para in enumerate(paralist):
    if len(str.split(para)) < 512:
        newparalist.append(para)


# Split the sentence into tokens.
tokenized_text = [tokenizer.tokenize(para) for para in newparalist]

# Map the token strings to their vocabulary indeces.
indexed_tokens = [tokenizer.convert_tokens_to_ids(para) for para in tokenized_text]

# segment IDs ... i have no idea what this means PLEASE CHECK UP ONTHIS

segments_ids = [1] * len(tokenized_text)

# Convert inputs to PyTorch tensors
tokens_tensors = []
segments_tensor = torch.tensor([segments_ids])

for para in indexed_tokens:

    tokens_tensor = torch.tensor([para])
    tokens_tensors.append(tokens_tensor)

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

# calculate sentence embedding for every paragraph

embedding = []

for id,tokens_tensor in enumerate(tokens_tensors):

    with torch.no_grad():
        segments_tensor = torch.tensor([1] * len(tokens_tensor))
        encoded_layers, _ = model(tokens_tensor, segments_tensor)

    token_embeddings = torch.stack(encoded_layers, dim=0)
    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    # Swap dimensions 0 and 1.
    token_embeddings = token_embeddings.permute(1,0,2)

    token_vecs = encoded_layers[11][0]

    # Calculate the average of token vectors
    sentence_embedding = torch.mean(token_vecs, dim=0)
    embedding.append(sentence_embedding)

    print(id/len(tokens_tensors))


with open('pickles/bertified_lol.pkl', 'wb') as f:
    pickle.dump([newparalist, embedding], f)

