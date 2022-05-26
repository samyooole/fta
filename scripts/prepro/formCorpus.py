"""
Call this function if you want to refresh the FTA corpus in text and receive in nltk format!
"""


import os
from nltk.corpus.reader.plaintext import PlaintextCorpusReader


def givemeCorpus():
    corpusdir = 'text/' # Directory of corpus.

    newcorpus = PlaintextCorpusReader(corpusdir, '.*')
    return newcorpus