from concurrent.futures import process
import os
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
import re
from itertools import chain
import pandas as pd
import numpy as np
from text_preprocessing import to_lower, check_spelling, expand_contraction, remove_number, remove_special_character, remove_punctuation, remove_whitespace, normalize_unicode, stem_word, remove_stopword, remove_single_characters, remove_url
import pickle

myFilters = [to_lower, expand_contraction, remove_special_character, remove_whitespace, normalize_unicode, remove_url]


def processor(func, text):
    output_list = []
    for i, item in enumerate(text): # so i can keep track of progress
        x=func(item)
        output_list.append(x)
        if round(i/len(text) * 100) % 5 == 0:
            print(i/len(text))

    return output_list

def processmyFilters(text, myFilters):
    startofprocess=text
    for filter in myFilters:
        endofprocess = filter(startofprocess)
        startofprocess = endofprocess
    return startofprocess


def givemeCorpus():
    """
    gives a standard NLTK corpus
    """
    corpusdir = 'text/' # Directory of corpus.

    newcorpus = PlaintextCorpusReader(corpusdir, '.*')
    return newcorpus

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf_8') as f:
        return f.read()

def givemeDocuments(corpusfolder):
    """
    runs through a corpus folder (with possible subfolders) to get a list of long strings per file
    """
    
    list_to_return = []
    
    path = os.getcwd()
    os.chdir(corpusfolder)
    for folder in os.listdir():
        if folder.startswith("."): # assumption is that management files that we want to ignore begin with .
            continue
        os.chdir(folder)
        for file in os.listdir():
            if file.endswith(".txt"):
                file_path = file
            else:
                continue
  
            # call read text file function
            addition = read_text_file(file_path)
            list_to_return.append(addition)
        os.chdir(os.path.dirname(os.getcwd()))
    os.chdir(os.path.dirname(os.getcwd()))
    return list_to_return

def splitParasbynewline(text):
    """
    takes in a text and splits it into paragraphs
    - brute force assumption: double space equals a paragraph break --> can we find a way to delimit by multiple conditions?
    - implicit: handling newlines using python standard https://stackoverflow.com/questions/16566268/remove-all-line-breaks-from-a-long-string-of-text
    """
    mid = re.split(r'(?: \n){2,}', text, flags=re.DOTALL) # I wish I could find a way to regex a string that starts with \n and ends with \n. this would be more general. but this works for now.
    mid = [item.replace("\n", " ") for item in mid]
    
    return mid

def splitParasbyArticle(text):
    """
    takes in a text and splits it into paragraphs
    - assumes the header "Article X" sufficiently divides for our purposes
    - add capturing group () to get the article name. if you want easy access next time
    """
    mid = re.split(r"\narticle \d.*?\n", text, flags= re.DOTALL | re.I)
    mid = [re.split(r"\narticle\d.*?\n", item, flags=re.DOTALL | re.I) for item in mid]
    mid = list(chain(*mid))
    mid = [re.split(r"\n\d\..*?\n", item, flags=re.DOTALL) for item in mid]
    mid = list(chain(*mid))

    output = [item.replace("\n", " ") for item in mid]
    return output

def splitParasbyArticlewithTagging(text):
    """
    returns pandas df
    """

    mid = re.split(r"\narticle \d.*?\n", text, flags= re.DOTALL | re.I)
    mid = [re.split(r"\narticle\d.*?\n", item, flags=re.DOTALL | re.I) for item in mid]

    # get chapter name
    chapter_noisy = mid[0][0]
    try:
        toparse = re.split(r"\n(?:annex|chapter) \d.*?\n", chapter_noisy, flags = re.DOTALL | re.I)[1] # get characters after the split
        chapter = toparse.replace("\n", "").strip().lower()
    except:
        chapter=np.nan
    

    # catch the any n words before the first newline. we assume this describes the title of the article

    mid = list(chain(*mid))
    mid.pop(0)

    updlist =[]

    for item in mid:
        spl = re.split("\n \n", item, flags= re.DOTALL | re.I) # more stringent, but makes the overall assumption that after each article title name we have a double newline
        article = spl[0].replace("\n", "").strip().lower()
        spl.pop(0)
        spl = "".join(spl).strip()

        spl = re.split(r"\n\d\..*?\n", spl, flags=re.DOTALL)
        spl = [re.split(r" \d\. ", item, flags=re.DOTALL) for item in spl]
        spl=list(chain(*spl))

        for idx, clause in enumerate(spl):
            clause = clause.replace("\n", "")
            clause = processmyFilters(text=clause, myFilters=myFilters)
            updlist.append([chapter, article, idx+1, clause]) #[chapter, article no., clause no., clause text ]

    df = pd.DataFrame(updlist, columns= ['chapter', 'article', 'clauseno', 'text'])

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
        df=df[['chapter', 'article', 'clauseno', 'text']]

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
        list_to_return[index][1] = splitParasbyArticle(feedtext)

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

    for item in list_to_return:
        feedtext = item[1]
        dftoupdate = splitParasbyArticlewithTagging(feedtext)
        clausedf=clausedf.append(dftoupdate)
    return clausedf

def getUnderstander(garble, legible):
    """
    returns a dictionary {tuple of words in a para, post-treatment: what the original text said}
    - basically we need this so that you can see, pre-treatment, what the text originally said so that it is more comprehensible to human eyes
    """
    paralist=[]
    for items in legible:
        paras = items[1]
        paralist.extend(paras)

    understander = {}
    for id, item in enumerate(garble):
        realtext = paralist[id]
        garbletext = tuple(item)
        understander.update({garbletext: realtext})

    return understander


def getKMiddle(input_list, K):
    
    # computing strt, and end index 
    strt_idx = (len(input_list) // 2) - (K // 2)
    end_idx = (len(input_list) // 2) + (K // 2)
    
    # slicing extracting middle elements
    res = input_list[strt_idx: end_idx + 1]

    return res