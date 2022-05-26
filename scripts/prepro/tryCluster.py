
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from nltk.corpus.reader.plaintext import PlaintextCorpusReader

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf_8') as f:
        return f.read()

def givemeDocuments():
    list_to_return = []
    
    path = 'C:\\Users\\Samuel\\fta'
    os.chdir('text')
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


def givemeCorpus():
    corpusdir = 'text/' # Directory of corpus.

    newcorpus = PlaintextCorpusReader(corpusdir, '.*')
    return newcorpus

# let us try vectorizing by paragraphs

text = givemeCorpus()

list_of_paras = text.paras()

vectorizer = TfidfVectorizer()

df = vectorizer.fit_transform(text)