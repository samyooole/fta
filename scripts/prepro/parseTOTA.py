import xml.etree.ElementTree as ET
import pandas as pd
import os
import re
from itertools import chain

df = pd.DataFrame()
lol=[]

# Passing the path of the
# xml document to enable the
# parsing process

def getelementfromHead(elementHead, elementName):
    items = elementHead.items()
    titleList = [item[0] for item in items]
    whereelement = titleList.index(elementName)
    returntext = items[whereelement][1]

    return returntext

for xml in os.listdir('tota'):
    tree = ET.parse('tota/'+xml)
    root = tree.getroot()
    meta = root[0]

    # get meta details of the fta
    """
    fta name format for tota is <full country name> - <full country name>
    """
    fta_name = meta.find('name').text
    rta_id = meta.find('wto_rta_id').text
    parties = [party.text for party in meta.find('parties').findall('partyisocode')] # returns a list
    parties = set(parties) # convert to set because the order of parties does not matter
    toa =  meta.find('type').text# type of agreement
    dif = meta.find('date_into_force').text

    """
    do a quick check of the meta: if it is non-english, we do not want it
    """
    lang = meta.findall('language')[0]
    if lang.text != 'en':
        continue

    body = root[1]
    for chapter in body:
        if 'name' not in chapter.keys():
            continue # we are not interested if a chapter doesn't have a name. for now - it's not useful for determining clause categories
        else:
            chapter_name = getelementfromHead(chapter, 'name')

        for article in chapter:
            if 'name' not in article.keys():
                continue # we are not interested if an article doesn't have a name. for now - it's not useful for determining clause categories
            article_name = getelementfromHead(article, 'name')
            article_text = article.text
            
            lol.append([fta_name, rta_id, parties, toa, dif, chapter_name, article_name, article_text])

df = pd.DataFrame(lol)

df.columns=['fta_name', 'rta_id', 'parties', 'toa', 'dif', 'chapter', 'article', 'text']

"""
divide fta articles into constituent clauses (marked by clause numbers). eg.

Article 3 
Transparency

1. Each Party shall promptly publish...

2. Following the end of the ...
"""

# clean out tags like \t \n and \xa0
text_cleaned = [item.replace('\t', '').replace('\n', '').replace('\xa0',' ') for item in df['text'] ]

df['text'] = text_cleaned


"""
split each article according to clause numbers, then explode down rows
"""

text=df['text']
newtext=[]
newrange=[]
for article in text:
    spl = re.split(r"\. +\d+\.", article, flags=re.DOTALL | re.I)
    spl = [clause.strip() for clause in spl]
    spl = [clause.strip("1. ") for clause in spl]

    rangeadd = list(range(1, len(spl)+1))
    if len(spl) == 1:
        rangeadd=[1]

    newrange.append(rangeadd)
    newtext.append(spl)

clauseno = list(chain(*newrange))

df['text'] = newtext

df = df.explode('text').reset_index()

df['clauseno'] = clauseno

df=df.drop('index', axis=1)

df.to_csv('scripts/prepro/totafta.csv', index=False)