# some pre-steps for mac.... cry
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download()

import sys
sys.path.append("scripts/prepro/")
from corpusManagement import getcorpusbyParas
from text_preprocessing import to_lower, check_spelling, expand_contraction, remove_number, remove_special_character, remove_punctuation, remove_whitespace, normalize_unicode, stem_word, remove_stopword, remove_single_characters, remove_url
import pickle

meta = getcorpusbyParas('text')

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

# check spelling - only do once you're certain everything is swee. very slow because it references a dictionary for every single word
# text_after_spellcheck = processor(check_spelling, text_after_to_lower)

# remove white spaces
text = processor(remove_whitespace, text)

# remove numbers (will we need figures and numbers at some point down? not for text classification at least)
text = processor(remove_number, text)

text = processor(remove_special_character, text)

#text = processor(remove_single_characters, text)

text = processor(normalize_unicode, text)

# remove punctuation last because it causes many issues
"""
re-wrote text_preprocessing fork to replace punc w a whitespace instead of nothing
"""
#text = processor(remove_punctuation, text)

#text = processor(remove_stopword, text)

# problem: we have some URLs that are broken by newlines, and thus cannot be caught by the above function. our band-aid solution for now is to get rid of any words greater than 20 characters (this is very rare in the english language). I make a judgement call that real words lost to this are exceedingly few as compared to accidental url concatenations

text = [ [word for word in para.split() if len(word) <= 20] for para in text]

text = [" ".join(para) for para in text]

#stemmed_text = processor(stem_word, text)

# save as pickle, takes a long time
with open('pickles/fullytreated_corpus.pkl', 'wb') as f:
    pickle.dump(text, f)

with open('pickles/paralist.pkl', 'wb') as f:
    pickle.dump(paralist, f)


# test pickle loading
file = open('pickles/fullytreated_corpus.pkl', 'rb')
new_pp = pickle.load(file) 


