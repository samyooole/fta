import spacy
import pandas as pd
from spacy import displacy
import networkx as nx
from spacy.matcher import Matcher
from itertools import chain, compress
from numpy import argmin
from pathlib import Path
import re

"""
Core idea:
- clauses surround a single action to be taken
--> eg. "Japan must give $5 to the US", the core action being "give $5", which is the core economic benefit that we want to capture

- however, clauses can get unwieldy, eg.
--> eg. "The Responding Party shall immediately acknowledge receipt of the request by way of notification to all Parties, indicating the date on which the request was received"
--> in this case, we might want to catch "immediately acknowledge receipt of the request", then link it to subject "Responding party" and object "all Parties"
--> then we leave "indicating the date..." as a clarifier action, rather than the core action

- therefore we do spacy/dependency_parsing -> networkx, which allows us to find the level of centrality, or rather the 'source'ness of each node. this will allow us to find the eldest verb head, which we theorize should correspond to the core action. other verb heads will give us clarifier actions

- this has already been done in the context of 'shortest dependency path'

///////
to work on in future:
-rules based splitter of lists, as delimited by (x) --> which then draws dependencies iteratively
"""



#load nlp object
nlp = spacy.load("en_core_web_sm")

"""
before moving forward, do some bespoke cleaning. though arguably this should have been done at the prepro stage, and so we will have to do a bit of cleanup to shift this to the front. will also affect models trained in between?
"""

# remove "-" because it seems to mess stuff up. read as compound word
df = pd.read_csv('core_excels/totaPlus_bypass.csv')
df = df[df['isSubstantive_predict'] == 1]
df=df.reset_index(drop=True)

df['text'] = df.text.apply(lambda x: x.replace("-", "").strip())


# split by periods, then explode. this is because clauses are sometimes composed of multiple sentences, which generally correspond to multiple core actions

df['text'] = df.text.apply(lambda x: x.split("."))
df['text'] = df.text.apply(lambda x: [item for item in x if item != ''])

df = df.explode('text')
df=df.drop('Unnamed: 0', axis=1)
df=df.reset_index(drop=True)
df['text'] = df.text.apply(lambda x: x.strip())

# try to get rid of all (x) to see if it fits the dependency parser model better
newtextfull=[]
for text in df.text:
    newtext=re.sub(r'\([a-z]{1}\)', '', text)
    newtext=re.sub(' {2,}', ' ', newtext)
    newtextfull.append(newtext)


####################

lol = []
for idx, text in enumerate(newtextfull):
    doc = nlp(text)

    """
    first, identify the head verb, which gives the core action
    """
    core_action_dict = [token for token in doc.to_json()['tokens'] if token['dep'] == 'ROOT' ]
    if core_action_dict == []:
        df=df.drop(idx)
        continue
    core_action = doc[core_action_dict[0]['id']]

    if (core_action.pos_ != 'VERB') & (core_action.pos_ != 'AUX'):
        df=df.drop(idx)
        continue

    """
    second, we identify the subject and object noun chunk with respect to the core action.

    - this is slightly complicated
    - we have to first identify the verb phrase, usually <part>* ? <adv>* <verb> <adv>* <prep>*
    - then, look for the children of all the parts of the verb phrase in order to locate the relevant 

    """

    # definining hierarchies for subject and object families
    objfam = ['dobj', 'pobj', 'iobj'] # direct object, indirect object, object of preposition
    subjfam = ['nsubj', 'nsubjpass', 'csubj', 'csubjpass' ] #nominal subject (passive), clausal subject (passive)

    # try doing the first descendant's parent method

    # first form the dependency tree in networkx
    edges=[]
    for token in doc:
        # FYI https://spacy.io/docs/api/token
        for child in token.children:
            edges.append(('{0}-{1}'.format(token.lower_,token.i),
                        '{0}-{1}'.format(child.lower_,child.i)))

    graph = nx.Graph(edges)

    # find quickest descendant subject

    descendant_dep = [des.dep_ for des in core_action.subtree] 
    descendants = [des for des in core_action.subtree]
    elig_subj = [des for des in descendants if des.dep_ in subjfam]
    elig_obj = [des for des in descendants if des.dep_ in objfam]

    # some more bespoke rules? --> encode in a proper function
    """
    we need some tie breakers when calculating the graph distance
    - basically we want stuff like 'forum' and not 'for the purposes of', and the theory anyway is that we can disqualify things that we know from a rules-based approach will definitely disqualify what we don't want
    - the following two lines disqualify subjects and objects that are preceded and succeeded by an ADP (adposition)
    """
    #elig_subj = list(compress(elig_subj, ['prep' not in [child.dep_ for child in subj.children] for subj in elig_subj]))
    #elig_obj = list(compress(elig_obj, ['prep' not in [child.dep_ for child in obj.children] for obj in elig_obj]))

    #elig_subj = [subj for subj in elig_subj if subj.head.pos_ != 'ADP']
    
    #elig_obj =  [obj for obj in elig_obj if obj.head.pos_ != 'ADP']
    

    if (elig_subj == []) | (elig_obj == []):
        df=df.drop(idx)
        continue

    elig_subj_formatted = ['{0}-{1}'.format(item.lower_, item.i) for item in elig_subj]
    core_action_formatted = '{0}-{1}'.format(core_action.lower_, core_action.i)

    lolengths = [nx.shortest_path_length(graph, source = item, target = core_action_formatted) for item in elig_subj_formatted]
    graph_distance_vote = argmin(lolengths)

    subj = elig_subj[graph_distance_vote]
    subjw=subj

    # find quickest descendant object

    elig_obj_formatted = ['{0}-{1}'.format(item.lower_, item.i) for item in elig_obj]
    core_action_formatted = '{0}-{1}'.format(core_action.lower_, core_action.i)

    lolengths = [nx.shortest_path_length(graph, source = item, target = core_action_formatted) for item in elig_obj_formatted]

    obj = elig_obj[argmin(lolengths)] # how to handle complicated sentence structures
    objw=obj

    """
    we also have a field for clausal complements
    """

    complement = [child for child in core_action.children if (child.dep_ == 'ccomp') | (child.dep_ == 'dative')]
    if complement == []:
        complement = ""
    else:

        complement = " ".join([branch.text for branch in complement[0].subtree])
    # to get the full chunk, we simply take the full subtree of the subj and obj respectively
    subj = " ".join([branch.text for branch in subj.subtree]) 
    obj = " ".join([branch.text for branch in obj.subtree]) 

    # get the core verb PHRASE rather than simply the core verb word
    # establish verb phrase. part credits to mdmjsh on stackoverflow
    pattern=[
    {'POS': 'AUX', 'OP': '*'},
    {'POS': 'PART', 'OP': '*'},
    {'POS': 'ADV', 'OP': '*'},
    {'POS': 'VERB', 'OP': '+'}, # the root can be either a verb or an aux ('is')
    {'POS': 'ADP', 'OP': '*'},
    ]

    matcher = Matcher(nlp.vocab)
    matcher.add('verb-phrases', [pattern])
    doc = nlp(text)
    matches = matcher(doc)
    spans = [doc[start:end] for _, start, end in matches] 

    list_of_elig = [span for span in spans if core_action.i in [spa.i for spa in span]]

    if len(list_of_elig) == 0:
        pattern=[
        {'POS': 'ADV', 'OP': '*'},
        {'POS': 'PART', 'OP': '*'},
        {'POS': 'AUX', 'OP': '+'}, # the root can be either a verb or an aux ('is')
        {'POS': 'ADP', 'OP': '*'}]

        matcher = Matcher(nlp.vocab)
        matcher.add('verb-phrases', [pattern])
        doc = nlp(text)
        matches = matcher(doc)
        spans = [doc[start:end] for _, start, end in matches] 

        list_of_elig = [span for span in spans if core_action.i in [spa.i for spa in span]]
    list_of_elig = [str(x) for x in list_of_elig]
    core_verb_phrase = max(list_of_elig,key=len)

    # experimental: if a root verb has an xcomp child, pick that child as the core action instead
    derp  = [child for child in core_action.children if child.dep_ == 'xcomp']
    if derp != []:
        core_action = derp[0]

    to_append = [subj, core_action, obj, complement]
    lol.append(to_append)

df=df.reset_index(drop=True)

checkdf=pd.concat([df[0:len(lol)], pd.DataFrame(lol, columns = ['subj', 'coreact', 'obj', 'compl'])], axis=1)

checkdf.to_csv('checkdf.csv')



html = displacy.render(doc, style='dep',page=True)
svg = displacy.render(doc, style="dep")
output_path = Path("sentence.svg")
output_path.open("w", encoding="utf-8").write(svg)

## networkx
# Load spacy's dependency tree into a networkx graph
edges = []
for token in doc:
    for child in token.children:
        edges.append(('{0}'.format(child.i),
                    '{0}'.format(token.i),
                      ))


# also create a dictionary that summarizes the identities and relations

G = nx.DiGraph(edges).reverse()





########## OTHER WAYS OF ESTABLISHING VERB PHRASE


    # establish verb phrase. part credits to mdmjsh on stackoverflow
    pattern=[
    {'POS': 'ADV', 'OP': '*'},
    {'POS': 'PART', 'OP': '*'},
    {'POS': 'VERB', 'OP': '+'}, # the root can be either a verb or an aux ('is')
    {'POS': 'ADP', 'OP': '*'},
    {'POS': 'AUX', 'OP': '*'}]

    matcher = Matcher(nlp.vocab)
    matcher.add('verb-phrases', [pattern])
    doc = nlp(text)
    matches = matcher(doc)
    spans = [doc[start:end] for _, start, end in matches] 

    list_of_elig = [span for span in spans if core_action.text in span.text]

    if len(list_of_elig) == 0:
        pattern=[
        {'POS': 'ADV', 'OP': '*'},
        {'POS': 'PART', 'OP': '*'},
        {'POS': 'AUX', 'OP': '+'}, # the root can be either a verb or an aux ('is')
        {'POS': 'ADP', 'OP': '*'}]

        matcher = Matcher(nlp.vocab)
        matcher.add('verb-phrases', [pattern])
        doc = nlp(text)
        matches = matcher(doc)
        spans = [doc[start:end] for _, start, end in matches] 

        list_of_elig = [span for span in spans if core_action.text in span.text]

    core_verb_phrase = max(list_of_elig)
    # NOTEEEEE there is still the matter of counting negation - negspacy, look later
    

    cvp_children = [ [child for child in token.children] for token in core_verb_phrase]
    cvp_children = list(chain(*cvp_children))

    children = [child for child in cvp_children]
    subj = [child for child in cvp_children if child.dep_ in subjfam][0]
    obj = [child for child in cvp_children if child.dep_ in objfam][0]