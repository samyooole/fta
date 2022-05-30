from corpusManagement import getcorpusbyParas
import pickle


text = getcorpusbyParas('text')

# some quick manual cleaning to reduce the # of rows
text = [item for item in text if item != '']