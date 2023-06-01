import sys
sys.path.append("scripts/prepro/")
from corpusManagement import splitParasbyArticlewithTagging, read_text_file, splitParasbyArticle, implicitCleaning
import os
import pandas as pd
from text_preprocessing import to_lower, check_spelling, expand_contraction, remove_number, remove_special_character, remove_punctuation, remove_whitespace, normalize_unicode, stem_word, remove_stopword, remove_single_characters, remove_url


os.chdir('cleantext/ksfta')

fulldf=pd.DataFrame()

for file in os.listdir():
    text = read_text_file(file)
    label = file[:-4]
    splitted = splitParasbyArticle(text)
    df=pd.DataFrame(splitted)
    df['label'] = label
    df.columns = ['text', 'catlabel']
    fulldf=fulldf.append(df)

os.chdir(os.path.dirname(os.getcwd()))
os.chdir('ussfta')

for file in os.listdir():
    text = read_text_file(file)
    label = file[:-4]
    splitted = splitParasbyArticle(text)
    df=pd.DataFrame(splitted)
    df['label'] = label
    df.columns = ['text', 'catlabel']
    fulldf=fulldf.append(df)

text = fulldf['text']

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
text = processor(to_lower, text)

# remove urls!!!! very important
text = processor(remove_url, text)

# expand contractions
text= processor(expand_contraction, text)

# remove white spaces
text = processor(remove_whitespace, text)

text = processor(remove_special_character, text)

text = processor(normalize_unicode, text)

# problem: we have some URLs that are broken by newlines, and thus cannot be caught by the above function. our band-aid solution for now is to get rid of any words greater than 20 characters (this is very rare in the english language). I make a judgement call that real words lost to this are exceedingly few as compared to accidental url concatenations

#text = [ [word for word in para.split() if len(word) <= 20] for para in text]

#text = [" ".join(para) for para in text]


fulldf['text'] = text

os.chdir(os.path.dirname(os.getcwd()))
os.chdir(os.path.dirname(os.getcwd()))

fulldf.to_csv("moretrainingdata1.csv")

labeldf=pd.read_csv('catTagging.csv')

fulldf=fulldf.merge(labeldf, left_on = 'catlabel', right_on = 'area', how='left')

fulldf[['text', 'catlabel_y']]

fulldf.to_csv('moretrainingdata1.csv')