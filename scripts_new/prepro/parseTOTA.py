import xml.etree.ElementTree as ET
import pandas as pd
import os
import re
from itertools import chain
from pandas.core.common import flatten
import pycountry as cty

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

for xml in os.listdir('tota_texts'):
    tree = ET.parse('tota_texts/'+xml)
    root = tree.getroot()
    meta = root[0]

    # get meta details of the fta
    """
    fta name format for tota is <full country name> - <full country name>
    """
    fta_name = meta.find('name').text
    rta_id = meta.find('wto_rta_id').text
    parties = [party.text for party in meta.find('parties').findall('partyisocode')] # returns a list
    if None in parties:
        parties = None
    else:

        parties = set(parties) # convert to set because the order of parties does not matter
    
    toa =  meta.find('type').text# type of agreement
    dif = meta.find('date_into_force').text
    din = meta.find('date_inactive').text
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
            
            lol.append([fta_name, rta_id, parties, toa, dif, din, chapter_name, article_name, article_text])

df = pd.DataFrame(lol)

df.columns=['fta_name', 'rta_id', 'parties', 'toa', 'dif', 'din', 'chapter', 'article', 'text']

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

"""
get empty party details from the RTA dataset
"""
rta = pd.read_excel('other_source/wtorta.xlsx')
rta = rta[['RTA ID', 'Original signatories']]
rta['RTA ID'] = rta['RTA ID'].apply(lambda x: float(x))
df = df.dropna(subset='rta_id')
df.rta_id = df.rta_id.apply(lambda x: float(x))

df = df.merge(rta, how='left', left_on = 'rta_id', right_on = 'RTA ID')

"""
compress into the empty slots
"""

# create a dictionary. {RTA name : ISO 3166 level code}

df['partiess'] = df.apply(lambda x: x['Original signatories'].split("; ") if pd.notnull(x['Original signatories']) else x['Original signatories'], axis=1)

subdf = df[df['parties'].isnull()]

# get unique country values. i <3 russian doll functions

countrynames = list(set(list(chain(*list(subdf.partiess)))))

sf = cty.countries.search_fuzzy

unfoundnames = []
ctydict = {}
for country in countrynames:
    try:
        iso = sf(country)[0].alpha_3
        ctydict.update({country:iso})
    except:
        unfoundnames.append(country)


ctydict.update(
    {'Türkiye': 
    
    sf('turkey')[0].alpha_3}
)

ctydict.update(
    {'Faeroe Islands': 
    
    sf('faroe islands')[0].alpha_3}
)

ctydict.update(
    {'UNMIK/Kosovo': 
    
    sf('kosovo')[0].alpha_3}
)


subdf['partiesss'] = subdf.apply(lambda x: set(list(flatten([ ctydict[party] for party in x['partiess'] ]))) if type(x['partiess']) is list else x['partiess'], axis=1)

subdf = subdf.drop_duplicates('fta_name')
subdf = subdf[['rta_id', 'partiesss']]

df=df.merge(subdf, how='left')

df['parties']=df.apply(lambda x: x['partiesss'] if pd.isnull(x['parties']) else x['parties'], axis=1)

df[df.parties.isnull()]

df=df.drop(['Original signatories', 'partiess', 'partiesss'], axis=1)

# some incomplete difs in tota
subdf = df[df['dif'].isnull()]
rta = pd.read_excel('other_source/wtorta.xlsx')
rta = rta[['RTA ID', 'Date of Entry into Force (G)']]
df = df.merge(rta, how='left')

df['dif'] = df.apply(lambda x: x['Date of Entry into Force (G)'] if pd.isnull(x['dif']) else x['dif'], axis=1)

df = df.drop(['RTA ID', 'Date of Entry into Force (G)'], axis=1)

df.to_csv('other_source/totafta.csv', index=False)