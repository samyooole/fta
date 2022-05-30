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


from sklearn.feature_extraction.text import TfidfVectorizer
from corpusManagement import getcorpusbyParas
from text_preprocessing import to_lower, check_spelling, expand_contraction, remove_number, remove_special_character, remove_punctuation, remove_whitespace, normalize_unicode, stem_word, lemmatize_word, preprocess_text, remove_stopword, remove_single_characters
import pickle

text = getcorpusbyParas('text')

# some quick manual cleaning to reduce the # of rows
text = [item for item in text if item != '']

"""
we perform the pre-processing one step by one step since it doesn't need to be automated. also, when combined together, the package text-preprocessing has very poor speed
"""
preprocess_functions = [to_lower, check_spelling, expand_contraction, remove_number, remove_whitespace, remove_single_characters]

def processor(func, text):
    output_list = []
    for i, item in enumerate(text): # so i can keep track of progress
        x=func(item)
        output_list.append(x)
        if (i/len(text) * 100) % 5 == 0:
            print(i/len(text))

    return output_list


# lower casing of text
text_after_to_lower = processor(to_lower, text)

# check spelling of text
text_after_expansion= processor(expand_contraction, text_after_to_lower)


# save as pickle, takes a long time
with open('fullytreated_corpus.pkl', 'wb') as f:
    pickle.dump(pp_text, f)

# test pickle loading
file = open('pickles/fullytreated_corpus.pkl', 'rb')
new_pp = pickle.load(file)