
from sklearn.feature_extraction.text import TfidfVectorizer
from corpusManagement import getcorpusbyParas
from text_preprocessing import to_lower, check_spelling, expand_contraction, remove_number, remove_special_character, remove_punctuation, remove_whitespace, normalize_unicode, stem_word, lemmatize_word, preprocess_text, remove_stopword, remove_single_characters
import pickle

text = getcorpusbyParas('text')

# some quick manual cleaning to reduce the # of rows
text = [item for item in text if item != '']

"""
choice of functions here will be very important - for instance, is it realistically feasible to catch numbers through nlp?
NOTE: the order somehow matters. source code isn't written well. because some functions like remove_number take in and spit out string, but others like remove_stopword will take in and spit out list? i think
"""
preprocess_functions = [to_lower, check_spelling, expand_contraction, remove_number, remove_special_character,remove_punctuation, remove_whitespace, normalize_unicode, remove_single_characters, remove_stopword, stem_word, lemmatize_word]

pp_text=[]
for i, item in enumerate(text): # so i can keep track of progress
    pp_text.append(preprocess_text(item, preprocess_functions))
    print(i/len(text))


# save as pickle, takes a long time
with open('fullytreated_corpus.pkl', 'wb') as f:
    pickle.dump(pp_text, f)

# test pickle loading
file = open('pickles/fullytreated_corpus.pkl', 'rb')
new_pp = pickle.load(file)