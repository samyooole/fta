import sys
sys.path.append("scripts/prepro/")
from corpusManagement import getcorpusbyParas, implicitCleaning
from text_preprocessing import to_lower, check_spelling, expand_contraction, remove_number, remove_special_character, remove_punctuation, remove_whitespace, normalize_unicode, stem_word, remove_stopword, remove_single_characters, remove_url
import pickle
from itertools import chain

meta = getcorpusbyParas('text') # temporary, move back once the new dict corpus style is done



# some quick manual cleaning to reduce the # of rows
for items in meta:
    paras = items[1]
    index = items[0]
    output = [para for para in paras if (para != '') & (not para.isspace())]
    
    meta[index][1] = output

with open('working_pickles/meta.pkl', 'wb') as f:
    pickle.dump(meta, f)

# create an index {doc_index: (start_para_index, end_para_index)} NO DELETION OF ENTRIES AFTER THIS. also form simple list of paras
start_index=0
docpara_dict={}
paralist=[]
for items in meta:
    paras = items[1]
    docindex = items[0]
    end_index=start_index+len(paras)-1
    
    docpara_dict.update({docindex: (start_index, end_index)})
    
    start_index=end_index+1
    paralist.extend(paras)


paralist = implicitCleaning(paralist)

"""
we perform the pre-processing one step by one step since it doesn't need to be automated. also, when combined together, the package text-preprocessing has very poor speed.
thoughts:
- punctuation could be useful?
"""

def processor(func, text):
    output_list = []
    for i, item in enumerate(text): # so i can keep track of progress
        x=func(item)
        output_list.append(x)
        if round(i/len(text) * 100) % 5 == 0:
            print(i/len(text))

    return output_list


# lower casing of text
text = processor(to_lower, paralist)

# remove urls!!!! very important
text = processor(remove_url, text)

# expand contractions
text= processor(expand_contraction, text)

# remove white spaces
text = processor(remove_whitespace, text)

text = processor(remove_special_character, text)

text = processor(normalize_unicode, text)

# problem: we have some URLs that are broken by newlines, and thus cannot be caught by the above function. our band-aid solution for now is to get rid of any words greater than 20 characters (this is very rare in the english language). I make a judgement call that real words lost to this are exceedingly few as compared to accidental url concatenations

text = [ [word for word in para.split() if len(word) <= 20] for para in text]

text = [" ".join(para) for para in text]

text = implicitCleaning(text) # second round of cleaning
 
# build df, output as both csv and list pickle
import pandas as pd
df=pd.DataFrame(text)# build dataframe here

df.columns=["text"]
df=df[df['text'].str.len() <30000] # get rid of absurdly long strings
df.to_csv('text_prelabel.csv')

#
text = df['text']
with open('pickles/fullytreated_corpus.pkl', 'wb') as f:
    pickle.dump(text, f)


# test pickle loading
file = open('pickles/fullytreated_corpus.pkl', 'rb')
new_pp = pickle.load(file) 