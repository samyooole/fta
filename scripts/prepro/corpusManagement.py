import os
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
import re


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
    
    path = 'C:\\Users\\Samuel\\fta'
    os.chdir(corpusfolder)
    for folder in os.listdir():
        if folder.endswith(".gitkeep"):
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
    return list_to_return

def splitParas(text):
    """
    takes in a text and splits it into paragraphs
    - brute force assumption: double space equals a paragraph break --> can we find a way to delimit by multiple conditions?
    """
    return re.split('  ', text)

def getcorpusbyParas(corpusfolder):
    """
    DANGER returns long list, don't call directly to interpreter
    - runs through a corpusfolder and returns a pure list of paragraphs
    """
    
    text = givemeDocuments(corpusfolder)

    newtext=[]
    for file in text:
        newtext.extend(splitParas(file))

    return newtext

