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
We simplify from the initial idea:

- We expect, with little loss of generality, that all FTA clauses are defined with respect to all parties
- therefore it is unimportant to figure out originator and destinator
- in a more generalizable model, we would be able to detect the origin and destination of an action, likely the subject of a machine learning tagging model (see https://aclanthology.org/2020.lrec-1.251.pdf)

- we can classify actions in two kinds:{
    self actions: 
        "Chile shall prohibit the importation, manufacture and sale of products with such geographical indications"
    
    mutual actions:
        "Each Party shall accord to the nationals of each other Party treatment no less favourable than it accords to its own nationals"
}

- we must first figure out what differentiates both actions. a mutual action has a separate destination, whereas a self action has a destination of the self

- 

"""



#load nlp object
nlp = spacy.load("en_core_web_sm")

"""
before moving forward, do some bespoke cleaning. though arguably this should have been done at the prepro stage, and so we will have to do a bit of cleanup to shift this to the front. will also affect models trained in between?
"""

# remove "-" because it seems to mess stuff up. read as compound word
df = pd.read_csv('core_excels/totaPlus_refinebranch.csv')
nonsub = df[df['isSubstantive_predict'] == 0]
df = df[df['isSubstantive_predict'] == 1]
df=df.reset_index(drop=True)

df['text'] = df.text.apply(lambda x: x.replace("-", "").strip())


# split by periods, then explode. this is because clauses are sometimes composed of multiple sentences, which generally correspond to multiple core actions
# WE DO NOT DO THIS. the costs of this rule outweigh the benefits
"""
df['text'] = df.text.apply(lambda x: x.split("."))
df['text'] = df.text.apply(lambda x: [item for item in x if item != ''])

df = df.explode('text')
df=df.drop('Unnamed: 0', axis=1)
df=df.reset_index(drop=True)
df['text'] = df.text.apply(lambda x: x.strip())
"""

"""
# explode based on (x). might as well just do a new effort lol
mid = df['text'].apply(lambda x: re.split(r'\([a-z]{1}\)', x))

lonewtext=[]
for txt in mid:
    if len(txt) == 1:
        toappend = txt
    else:
        mide = txt
        toappend = [mide[0] + item for item in mide if item != mide[0]]
    lonewtext.append(toappend)


df['text'] = lonewtext
df=df.explode('text')
df=df.reset_index(drop=True)
df['text'] = df.text.apply(lambda x: x.strip())
"""

####################

lol = []
for idx, text in enumerate(df.text):
    doc = nlp(text)

    """
    first, identify the head verb, which gives the core action
    """
    core_action_dict = [token for token in doc.to_json()['tokens'] if token['dep'] == 'ROOT' ]
    if core_action_dict == []:
        to_append = [None,None,None,None]
        lol.append(to_append)
        continue
    core_action = doc[core_action_dict[0]['id']]

    if (core_action.pos_ != 'VERB') & (core_action.pos_ != 'AUX'):
        to_append = [None,None,None,None]
        lol.append(to_append)
        continue


    """
    second, we identify the subject and object noun chunk with respect to the core action.

    - this is slightly complicated
    - we have to first identify the verb phrase, usually <part>* ? <adv>* <verb> <adv>* <prep>*
    - then, look for the children of all the parts of the verb phrase in order to locate the relevant 

    """

    # definining hierarchies for subject and object families
    objfam = ['dobj', 'pobj', 'iobj', 'nsubjpass'] # direct object, indirect object, object of preposition, passive subject (may be considered the destination of the action)
    subjfam = ['nsubj', 'acomp', 'cop'] #nominal subject DONT USE clausal subject 

    # try doing the first descendant's parent method

    # first form the dependency tree in networkx
    edges=[]
    for token in doc:
        # FYI https://spacy.io/docs/api/token
        for child in token.children:
            edges.append(('{0}-{1}'.format(token.lower_,token.i),
                        '{0}-{1}'.format(child.lower_,child.i)))

    graph = nx.Graph(edges)

    """
    we also have a field for clausal complements
    """

    complement = [child for child in core_action.children if (child.dep_ == 'ccomp') | (child.dep_ == 'dative')]
    if complement == []:
        complement = ""
    else:

        complement = " ".join([branch.text for branch in complement[0].subtree])



    
    # Mixed Quickest Descendant Method - for subj/obj identification

    core_action_subj = core_action
    
    xcomp_child  = [child for child in core_action.children if (child.dep_ == 'xcomp') & (child.pos_ == 'VERB')]
    if xcomp_child != []:
        core_action_obj = xcomp_child[0]
    else:
        core_action_obj = core_action
    
    descendants_subj = [des for des in core_action_subj.subtree]
    descendants_obj = [des for des in core_action_obj.subtree]

    elig_subj = [des for des in descendants_subj if des.dep_ in subjfam]
    elig_obj = [des for des in descendants_obj if des.dep_ in objfam]

   

    # some more bespoke rules? --> encode in a proper function
    """
    - theory: if I am the closest descendant to the root verb, I am the subject/object of the core action
    - there are many ties, which we have to break in some way
    - one possible way is to disqualify all such eligible subjects/objects if they are not a member of noun chunks
    """

    # idea, we disqualify potential objects and subjects where they are not a member of a noun chunk
    nchunks = [chunk for chunk in doc.noun_chunks]
    nchunks_id = [[word.i for word in chunk] for chunk in nchunks]

    
    nestedbool = [[subjw.i in chunk for chunk in nchunks_id] for subjw in elig_subj]

    subjboolfilter = []
    for idx, innerbool in enumerate(nestedbool):
        if True in innerbool:
            subjboolfilter.append(idx)

    elig_subj = [elig_subj[i] for i in subjboolfilter]

    nestedbool = [[objw.i in chunk for chunk in nchunks_id] for objw in elig_obj]

    objboolfilter = []
    for idx, innerbool in enumerate(nestedbool):
        if True in innerbool:
            objboolfilter.append(idx)

    elig_obj = [elig_obj[i] for i in objboolfilter]



    if (elig_obj == []):
        to_append = [None,None,None,None]
        lol.append(to_append)
        continue

    # for SUBJECT
    elig_subj_formatted = ['{0}-{1}'.format(item.lower_, item.i) for item in elig_subj]
    core_action_subj_formatted = '{0}-{1}'.format(core_action_subj.lower_, core_action_subj.i)
    

    lolengths = [nx.shortest_path_length(graph, source = item, target = core_action_subj_formatted) for item in elig_subj_formatted]
    
    if lolengths == []:
        subj = None
        subjw =subj
        subjNC = None
    else:
        graph_distance_vote = argmin(lolengths)
        subj = elig_subj[graph_distance_vote]
        subjw=subj

        boolbool = [subjw.i in chunk for chunk in nchunks_id]
        subjNC = list(compress(nchunks, boolbool))[0]
        subjNC = subjNC.text

        subj = " ".join([branch.text for branch in subjw.subtree]) 

    # for OBJECT

    elig_obj_formatted = ['{0}-{1}'.format(item.lower_, item.i) for item in elig_obj]
    core_action_obj_formatted = '{0}-{1}'.format(core_action_obj.lower_, core_action_obj.i)

    lolengths = [nx.shortest_path_length(graph, source = item, target = core_action_obj_formatted) for item in elig_obj_formatted]

    obj = elig_obj[argmin(lolengths)] # how to handle complicated sentence structures
    objw=obj
    
    # get subj and obj within their respective noun chunks

    

    
    boolbool = [objw.i in chunk for chunk in nchunks_id]
    objNC = list(compress(nchunks, boolbool))[0]

    # to get the full chunk, we simply take the full subtree of the subj and obj respectively
    
    obj = " ".join([branch.text for branch in objw.subtree]) 


    # get
    """
    ADVERBIAL CLAUSE MODIFIER
    - essentially an indicator for a conditional.

    illustrative example.

    "If it is impracticable to comply immediately with the obligation in Paragraph 1, the Responding Party shall have a reasonable period of time to do so"

    core action -> have
    object -> a reasonable period of time
    advcl modifier -> if it is impracticable to comply immediately with the obligation in Para 1

    """
    advclroot = [child for child in core_action.children if child.dep_ == 'advcl']
    advcllist=[]
    for root in advclroot:
        
        if root != None:
            advcl = " ".join([branch.text for branch in root.subtree])
        else:
            advcl = None

        advcllist.append(advcl)

    # get
    """
    ADJECTIVAL COMPLEMENT
    eg. Proceedings shall be confidential
    subj -> Proceedings
    core action -> shall be
    acomp -> confidential

    the idea is where a core_action_obj has an acomp child, get the acomp child


    """

    acomp_children = [child for child in core_action_obj.children if child.dep_ == "acomp"]

    acomplist = []
    for root in acomp_children:
        
        if root != None:
            advcl = " ".join([branch.text for branch in root.subtree])
        else:
            advcl = None

        acomplist.append(advcl)

    acomplist = "".join(acomplist)

    # experimental: if a root verb has an xcomp child, pick that child as the core action instead
    """
    illustrative example.

    "Japan is deemed to have violated the terms of the trade agreement..."

    root verb = 'deemed'
    'deemed' --xcomp child--> 'violated'

    """

    xcomp_child  = [child for child in core_action.children if (child.dep_ == 'xcomp') & (child.pos_ == 'VERB')]
    if xcomp_child != []:
        core_action = xcomp_child[0]


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
        {'POS': 'AUX', 'OP': '*'},
        {'POS': 'PART', 'OP': '*'},
        {'POS': 'ADV', 'OP': '*'},
        {'POS': 'AUX', 'OP': '+'}, # the root can be either a verb or an aux ('is')
        {'POS': 'ADP', 'OP': '*'},]

        matcher = Matcher(nlp.vocab)
        matcher.add('verb-phrases', [pattern])
        doc = nlp(text)
        matches = matcher(doc)
        spans = [doc[start:end] for _, start, end in matches] 

        list_of_elig = [span for span in spans if core_action.i in [spa.i for spa in span]]
    list_of_elig = [str(x) for x in list_of_elig]
    core_verb_phrase = max(list_of_elig,key=len)

    to_append = [subjNC, core_action.lemma_, objNC.text, acomplist, complement, advcllist, subj, core_verb_phrase, obj]
    lol.append(to_append)

df=df.reset_index(drop=True)

checkdf=pd.concat([df[0:len(lol)], pd.DataFrame(lol, columns = ['subjw','coreact', 'objw', 'objw_adjectival', 'clausal_complement', 'conditionals', 'subj', 'cvp', 'obj'])], axis=1)

nonsub[['subjw','coreact', 'objw', 'objw_adjectival', 'clausal_complement', 'conditionals', 'subj', 'cvp', 'obj']] = None

checkdf = pd.concat([checkdf, nonsub])

checkdf.to_csv('core_excels/totaPlus_dp.csv')



html = displacy.render(doc, style='dep',page=True)
svg = displacy.render(doc, style="dep")
output_path = Path("sentence.svg")
output_path.open("w", encoding="utf-8").write(svg)


#######3 DUMP



    