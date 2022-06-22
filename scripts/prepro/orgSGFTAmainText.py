import os
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
import re
from itertools import chain
import pandas as pd
import numpy as np
from text_preprocessing import to_lower, expand_contraction, remove_special_character, remove_whitespace, normalize_unicode, remove_url

myFilters = [to_lower, expand_contraction, remove_special_character, remove_whitespace, normalize_unicode, remove_url]


def processmyFilters(text, myFilters):
    startofprocess=text
    for filter in myFilters:
        endofprocess = filter(startofprocess)
        startofprocess = endofprocess
    return startofprocess


def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf_8') as f:
        return f.read()


def splitParasbynewline(text):
    """
    takes in a text and splits it into paragraphs
    - brute force assumption: double space equals a paragraph break --> can we find a way to delimit by multiple conditions?
    - implicit: handling newlines using python standard https://stackoverflow.com/questions/16566268/remove-all-line-breaks-from-a-long-string-of-text
    """
    mid = re.split(r'(?: \n){2,}', text, flags=re.DOTALL) # I wish I could find a way to regex a string that starts with \n and ends with \n. this would be more general. but this works for now.
    mid = [item.replace("\n", " ") for item in mid]
    
    return mid


def splitParasbyArticlewithTagging(text, spare_title, fta_name):
    """
    returns pandas df
    """

    # unicode nonsense: replace \xa0 with a space
    text = text.replace('\xa0', ' ')

    mid = re.split(r"\n *(?:rule|article) \d+ *?\n", text, flags= re.DOTALL | re.I)
    mid = [re.split(r"\n *(?:rule|article)\d+ *?\n", item, flags=re.DOTALL | re.I) for item in mid]
    mid = list(chain(*mid))
    mid = [re.split(r"\n *(?:rule|article) \d+\.\d+ *?\n", item, flags=re.DOTALL | re.I) for item in mid]
    mid = list(chain(*mid))
    mid = [re.split(r"\n *(?:rule|article) \d+\.\d+ *:", item, flags=re.DOTALL | re.I) for item in mid]
    mid = list(chain(*mid))
    mid = [re.split(r"\n *(?:rule|article) \d+ *:", item, flags=re.DOTALL | re.I) for item in mid]
    mid = list(chain(*mid))
    #mid = [re.split(r"\n+(?:rule|article)", item, flags=re.DOTALL | re.I) for item in mid]
    #mid = list(chain(*mid))
    #mid = [re.split(r"\. +(?:rule|article)", item, flags=re.DOTALL | re.I) for item in mid]
    #mid = list(chain(*mid))

    # get chapter name
    #mid = [elem.strip() for elem in mid if elem.strip()]
    chapter_noisy = mid[0]
    try:
        toparse = re.split(r"chapter", chapter_noisy, flags = re.DOTALL | re.I)[1] # get characters after the split
        
        toparse = toparse.replace("\n", "").strip().lower()
        chapter = re.sub('[^A-Za-z]+',' ', toparse)
        chapter = chapter.strip()
        mid.pop(0)
    except:
        chapter=spare_title
        mid.pop(0)
    

    # catch the any n words before the first newline. we assume this describes the title of the article
    

    updlist =[]

    for idx,item in enumerate(mid):
        
        """ first cut: read up till the first 1. for the article name"""
        splitsearch=re.match(".*?(?=[ \d]\. )", item, flags=re.DOTALL|re.I)

        if splitsearch is None:
            spl = re.split("\n", item, flags= re.DOTALL | re.I) # more stringent, but makes the overall assumption that after each article title name we have a double newline
            spl = [elem for elem in spl if elem.strip()]
            article = spl[0].replace("\n", "").strip().lower()
            
        else:
            article = splitsearch[0]
            spl = re.split("\n", article, flags= re.DOTALL | re.I)
            spl = [elem for elem in spl if elem.strip()]
            if spl == []:
                continue
            article = spl[0].strip().lower()
            

        article = re.sub('[^A-Za-z\-]+',' ', article)
        article = article.strip()
        
        spl= item.replace("\n", "")
        splitted = re.split(r"\. +\d+\.", spl, flags=re.DOTALL | re.I)
        splitted = [" ".join(clause.strip().split()) for clause in splitted]
        
        art_identifier=idx

        for idx, clause in enumerate(splitted):
            clause = processmyFilters(text=clause, myFilters=myFilters)
            updlist.append([fta_name,chapter, article, art_identifier, idx+1, clause]) #[chapter, article no., clause no., clause text ]

        

    df = pd.DataFrame(updlist, columns= ['fta_name','chapter', 'article', 'art_identifier', 'clauseno', 'text'])

    if df.empty == True:
        """
        to improve in the future: we may try to assume that any thing that does not follow the convention is a schedule of some form, so we can try to use this to apply the schedule parsing"""
        output = splitParasbynewline(text)
        output = [processmyFilters(item,myFilters) for item in output]
        df = pd.DataFrame(output)
        df.columns = ["text"]
        df['chapter'] = np.nan
        df['article'] = np.nan
        df['clauseno'] = np.nan
        df['fta_name'] = fta_name
        df['art_identifier'] = np.nan
        df=df[['fta_name','chapter', 'article', 'art_identifier', 'clauseno', 'text']]

    return df



def getcorpusbyParas(corpusfolder):
    """
    DANGER returns long list, don't call directly to interpreter
    - runs through a corpusfolder and returns list of lists ([1: text, 2: fta name, 3: sub fta name, 0: paragraph-wise index])
    - the purpose of getting the meta information is so that we have a standardized index from start to finish
    """
    
    list_to_return = []

    path = os.getcwd()
    os.chdir(corpusfolder)
    for folder in os.listdir():
        if folder.startswith("."): # assumption is that management files that we want to ignore begin with .
            continue
        os.chdir(folder)
        for file in os.listdir():
            
            
            if file == 'mothertext.pdf':
                continue
            
            if file.endswith(".txt"):
                file_path = file
            else:
                continue
  
            # call read text file function
            addition = read_text_file(file_path)
            list_to_return.append([len(list_to_return), addition, folder, file ])

        os.chdir(os.path.dirname(os.getcwd()))
    os.chdir(os.path.dirname(os.getcwd()))
    
    for item in list_to_return:
        index = item[0]
        feedtext = item[1]
        spare_title = item[-1]
        list_to_return[index][1] = splitParasbyArticle(feedtext, spare_title=spare_title)

    return list_to_return


def newgetCorpus(corpusfolder):
    
    list_to_return = []

    path = os.getcwd()
    os.chdir(corpusfolder)
    for folder in os.listdir():
        if folder.startswith("."): # assumption is that management files that we want to ignore begin with .
            continue
        os.chdir(folder)
        for file in os.listdir():
            
            
            if file == 'mothertext.pdf':
                continue

            if re.search('annex', file, flags= re.I) is not None:
                continue
            
            if file.endswith(".txt"):
                file_path = file
            else:
                continue
  
            # call read text file function
            addition = read_text_file(file_path)
            list_to_return.append([len(list_to_return), addition, folder, file ])

        os.chdir(os.path.dirname(os.getcwd()))
    os.chdir(os.path.dirname(os.getcwd()))
    
    clausedf = pd.DataFrame(columns=['chapter', 'article', 'clauseno', 'text'])

    for idx,item in enumerate(list_to_return):
        feedtext = item[1]
        spare_title = item[-1][:-4]
        fta_name= item[-2]
        dftoupdate = splitParasbyArticlewithTagging(feedtext, spare_title=spare_title, fta_name=fta_name)
        clausedf=clausedf.append(dftoupdate)
        print(idx/len(list_to_return))
    return clausedf



df=newgetCorpus('text')

df=df.reset_index()

df['assumed_article']=df['text'].apply(lambda x: re.match(".*?(?= [\d]\. )", x, flags=re.DOTALL|re.I)[0] if re.match(".*?(?= [\d]\. )", x, flags=re.DOTALL|re.I) is not None else "")

df['article'] = df[['article', 'clauseno', 'assumed_article']].apply(lambda x: x['assumed_article'] if (x['article'] == "") & (x['clauseno'] == 1) & (len(x['assumed_article'].split()) < 10) else x['article'], axis=1)

df=df.replace(r'^\s*$', np.nan, regex=True)




df['article'] = df.groupby(['fta_name', 'chapter', 'art_identifier']).article.ffill()


# clean up chapter names that contain section which provides additional useless information
df['chapter']=df['chapter'].apply(lambda x: re.split("section", x, flags=re.I)[0] if pd.notnull(x) else x)

"""
for some clauses, we may have such instances, where:

Article 1
Expropriation and Compensation

1. For the purposes of this agreement,...

may be interpreted as

'expropriation and compensation 1. for the purposes of this agreement'

So, we want to get rid of the leading title within the clause text itself, since that information has already been captured in the corresponding row.

We also want to get rid of the 1. for cleanliness purposes
"""

df=df.convert_dtypes()
df['article']=df.article.fillna('')
df['text']=df.text.fillna('')

# delete leading title
df.text = df[['article', 'text']].apply(lambda x: x['text'][len(x['article']):] if x['text'].startswith(x['article']) else x['text'], axis=1)

# delete leading 1.
df['text'] = df['text'].apply(lambda x: re.split(r'^ *\d\.', x, re.DOTALL)[-1])

"""
remove numbers in alphabetical form from chapter titles
"""

numbers=['one','two','three', 'four', 'five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen', 'twenty']
#df['chapter'].apply(lambda x: x.replace(number, '') if any(number in x for number in numbers) else x)
df=df.dropna(subset='chapter')
newchaplist=[]
for chapter in df['chapter']:
    for number in numbers:
        chapter=chapter.replace(number, '')
    
    newchaplist.append(chapter)

df['chapter'] = newchaplist


df=df.drop(["index","art_identifier", "assumed_article"], axis=1)

"""
attempt to split words that have a spelling error.

eg.
investmentprotection > investment protection
"""

import splitter
import enchant

print(enchant.list_dicts())

splitter.split('artfactory')
df.to_csv('tagged.csv')

d=enchant.Dict('en_US')