import os
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
import re
from itertools import chain
import pickle


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
    mid = re.split(' \n \n', text) # I wish I could find a way to regex a string that starts with \n and ends with \n. this would be more general. but this works for now.
    mid = [item.replace("\n", " ") for item in mid]
    mid = [re.split(r'\s{4,}', item) for item in mid] #arbitrary min length
    output = list(chain(*mid))
    

    return output

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