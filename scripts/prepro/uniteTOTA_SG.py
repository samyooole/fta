import pandas as pd
import pycountry as cty
from itertools import chain
from pandas.core.common import flatten
from datetime import datetime

date = datetime.date

"""
first, we load in the sg FTA dataset and bring it up to parity with tota.

to do this, we must get the RTA database. aka the Regional Trade Agreements database maintained by the World Trade Organization
"""

sg = pd.read_csv("scripts/prepro/sgfta.csv")
rta = pd.read_excel("scripts/prepro/wtorta.xlsx")
mapper = pd.read_csv("scripts/prepro/esgFTA_to_rta.csv")

rta = rta[['RTA ID', 'Type', 'Date of Entry into Force (G)', 'Date of Entry into Force (S)', 'Original signatories']]

sg=sg.merge(mapper, how='left', left_on='fta_name', right_on='esg_name')
sg=sg.merge(rta, how='left', left_on='rta_id', right_on = 'RTA ID')

# create a dictionary. {RTA name : ISO 3166 level code}

sg['parties'] = sg.apply(lambda x: x['Original signatories'].split("; ") if pd.notnull(x['Original signatories']) else x['Original signatories'], axis=1)

# get unique country values. i <3 russian doll functions

countrynames = list(set(list(chain(*list(sg.parties[sg.parties.notnull()])))))

sf = cty.countries.search_fuzzy

unfoundnames = []
ctydict = {}
for country in countrynames:
    try:
        iso = sf(country)[0].alpha_3
        ctydict.update({country:iso})
    except:
        unfoundnames.append(country)

# manually add the rest
unfoundnames

ctydict.update(
    {'European Free Trade Association (EFTA)': 
    
    [sf('ireland')[0].alpha_3, sf('liechtenstein')[0].alpha_3, sf('norway')[0].alpha_3, sf('switzerland')[0].alpha_3]}
)

ctydict.update(
    {'Hong Kong, China': 
    
    sf('hong kong')[0].alpha_3}
)

ctydict.update(
    {'Türkiye': 
    
    sf('turkey')[0].alpha_3}
)

ctydict.update(
    {'ASEAN Free Trade Area (AFTA)': 
    
    [sf('brunei')[0].alpha_3,
    sf('cambodia')[0].alpha_3,
    sf('indonesia')[0].alpha_3,
    'LAO',
    sf('malaysia')[0].alpha_3,
    sf('myanmar')[0].alpha_3,
    sf('philippines')[0].alpha_3,
    sf('singapore')[0].alpha_3,
    sf('thailand')[0].alpha_3,
    sf('vietnam')[0].alpha_3,
    ]
    
    }
)

ctydict.update(
    {'Gulf Cooperation Council (GCC)': 
    
    [sf('bahrain')[0].alpha_3,
    sf('kuwait')[0].alpha_3,
    sf('oman')[0].alpha_3,
    sf('qatar')[0].alpha_3,
    sf('saudi arabia')[0].alpha_3,
    sf('united arab emirates')[0].alpha_3,
    ]
    
    }
)

sg['parties'] = sg.apply(lambda x: set(list(flatten([ ctydict[party] for party in x['parties'] ]))) if type(x['parties']) is list else x['parties'], axis=1)

sg=sg.drop(['esg_name', 'RTA ID', 'Original signatories'], axis=1)


"""
how different are the entry into force dates of goods vs services agreements?
"""

sg.drop_duplicates('fta_name')[['fta_name','Date of Entry into Force (G)', 'Date of Entry into Force (S)']]

"""
there are some discrepancies for a few, but they are mostly the same. we take the goods date of entry into force for now. TO FIX IF WE WANT TO DISAMBIGUATE BETWEEN SERVICES AND GOODS
"""

sg['dif'] = sg['Date of Entry into Force (G)']

sg=sg.drop(['Date of Entry into Force (G)', 'Date of Entry into Force (S)'], axis=1)
sg=sg.rename({'Type': 'toa'}, axis=1)

# finally, map fta names to country sets where RTA has not identified them in their database yet. also, give bespoke placeholder rta ids

nonRTAs = pd.unique(sg[sg.rta_id.isnull()].fta_name)

ctydict.update(
{'rcep':

    [

    sf('brunei')[0].alpha_3,
    sf('cambodia')[0].alpha_3,
    sf('indonesia')[0].alpha_3,
    'LAO',
    sf('malaysia')[0].alpha_3,
    sf('myanmar')[0].alpha_3,
    sf('philippines')[0].alpha_3,
    sf('singapore')[0].alpha_3,
    sf('thailand')[0].alpha_3,
    sf('vietnam')[0].alpha_3, # asean    
        
    sf('australia')[0].alpha_3,
    sf('china')[0].alpha_3,
    sf('japan')[0].alpha_3,
    sf('new zealand')[0].alpha_3,
    sf('republic of korea')[0].alpha_3, # +5
    ]

}
)

ctydict.update(
{'slsfta':

    [

    sf('singapore')[0].alpha_3,
    sf('sri lanka')[0].alpha_3,
    ]

}
)
difdict={'rcep':'2022-01-01', 'slsfta':'2018-05-01'}

sg['parties']=sg.apply(lambda x: set(ctydict[x['fta_name']]) if x['fta_name'] in nonRTAs else x['parties'], axis=1 )


sg['dif'] = sg.apply(lambda x: difdict[x['fta_name']] if x['fta_name'] in nonRTAs else str(x['dif'].date()), axis=1 )

bespoke_rtadict = {'rcep':'new1', 'slsfta':'new2'}

sg['rta_id'] = sg.apply(lambda x: bespoke_rtadict[x['fta_name']] if x['fta_name'] in nonRTAs else str(int(x['rta_id'])), axis=1 )

pd.unique(sg.rta_id)

"""
read tota. remember that we have to deduplicate tota 
"""

tota = pd.read_csv('scripts/prepro/totafta.csv')
tota['rta_id'] = tota.apply(lambda x: str(int(x['rta_id'])) if pd.notnull(x['rta_id']) else x['rta_id'], axis=1)

tota_rtas = set(pd.unique(tota['rta_id']))
sg_rtas = set(pd.unique(sg.rta_id))
doublecounted = tota_rtas.intersection(sg_rtas) # if u need to find out what tota has already studied

"""
judgement call: for those which both tota and esg double count, we take the esg one
"""
isnotESG = tota.apply(lambda x: False if x['rta_id'] in doublecounted else True, axis=1)

tota_excl = tota[isnotESG]

"""
finally, we append ESG ftas to TOTA
"""

sg.append(tota_excl)

newtota = pd.concat([sg, tota_excl], axis=0).reset_index()

newtota.to_csv('newtota.csv')
