import pandas as pd
import sys
sys.path.append("scripts/model/")
from cleanandTransform import cleanandTransform

cat = cleanandTransform(filters = [], transformer_package='all-MiniLM-L12-v2')

df=pd.read_csv('core_excels/articlesAssigned.csv')
df=df[['chapter', 'chapter_cat', 'article', 'clauseno', 'text', 'fta_name', 'rta_id', 'toa', 'parties', 'dif','isSubstantive_predict', ]]

text = df.text.apply(lambda x: str(x))

cat.set_filters(['to_lower', 'remove_url','remove_special_character', 'normalize_unicode', 'remove_number', 'remove_punctuation', 'remove_stopword', 'stem_word', 'remove_whitespace'])
cat.init_text(text)
cat.process_text()

"""
active question: to remove non-substantive clauses or not?"""

## for one particular chapter

chapter = 'competition policy'
indexing = df['chapter_cat'] == chapter
indexing = list(indexing)

relevant_text = pd.Series(cat.current_text)[indexing]

# coerce relevant text
coerced_text = [clause.split() for clause in relevant_text]

### create bigrams, add them to the model (following from gensim example)
from gensim.models import Phrases

bigram = Phrases(coerced_text, min_count=20)
for idx, clause in enumerate(coerced_text):
    for token in bigram[clause]:
        if "_" in token:
            coerced_text[idx].append(token)


## create bag of words
from gensim.corpora import Dictionary

dictionary = Dictionary(coerced_text)
corpus = [dictionary.doc2bow(doc) for doc in coerced_text]

#HDP LDA (hierarchical)
from gensim.models import HdpModel
from numpy import argmax

hdp = HdpModel(corpus, dictionary)

argmaxtopics = []
for classi in hdp[corpus]:
    scores = [unit[1] for unit in classi]
    argmaxtopic = classi[argmax(scores)][0]
    argmaxtopics.append(argmaxtopic)

intelligible_text = df.text[indexing].reset_index()

sample = pd.concat([intelligible_text, pd.Series(argmaxtopics)], axis=1)

sample.to_csv('seeHDPLDA.csv')


























#######################################################################
# Train LDA model. (must specify no. of topics)
from gensim.models import LdaModel

# Set training parameters.
num_topics = 10
chunksize = 2000
passes = 20
iterations = 400
eval_every = None  # Don't evaluate model perplexity, takes too much time.

# Make an index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
)
