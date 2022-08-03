"""
Now that we have our totaPlus dataset, we wish to pull relevant data from the web to form a comprehensive trade dataset.

In particular, we aim towards a baseline trade gravity model with fixed effects.

time index = year (as at start of year)


"""

import pandas as pd
from itertools import combinations
from dask.dataframe import from_pandas




df = pd.read_csv("core_excels/totaPlus_MandatoryPredictHF.csv")

df['clause_id'] = df.apply(lambda x: x['chapter_cat'] + "_" + x['article_cat'],axis=1)

# drop clause_ids that directly reflect the relevant FTA?

find_singletons = df.groupby(['clause_id'])['fta_name'].nunique()


singletons = find_singletons[find_singletons == 1].index.to_list()

df = df[~df['clause_id'].isin(singletons)]

# filter out only the substantive clauses

df=df[df['isSubstantive_predict'] == 1]


"""
select relevant columns<article cat classification>"""

df=df[['parties', 'dif', 'din', 'clause_id']]

"""
df['clause_id'] = df.groupby(['chapter_cat', 'article_cat']).ngroup()
df=df.drop(['chapter_cat', 'article_cat'],axis=1)
df=df.drop_duplicates(['parties', 'clause_id']).reset_index(drop=True)


"""
#keep a keyer DF for clause_ids

"""
keydf = pd.read_csv('core_excels/totaPlus_MandatoryPredictHF.csv')
keydf=keydf[['parties', 'dif', 'din', 'chapter_cat', 'article_cat']]
keydf['clause_id'] = keydf.groupby(['chapter_cat', 'article_cat']).ngroup()
keydf = keydf[['chapter_cat', 'article_cat', 'clause_id']]
keydf['clause_id_verbose'] = keydf.apply(lambda x: x['chapter_cat'] + "_" + x["article_cat"],axis=1)

"""

"""
how many of this particular clause does an exporter-importer pair actively share at this point in time?
"""

import numpy as np
newdf = pd.DataFrame(np.repeat(df.values, 26, axis=0))
newdf.columns = df.columns

newdf['year'] = list(range(1995,2021)) * int(len(newdf) / 26)

df=newdf
del newdf

"""
specify whether a clause is in fact active for an exporter-importer pair

with the column clauseyear_dum
"""

df['dif'] = pd.to_datetime(df.dif)
df['din'] = df['din'].apply(lambda x: '2050' if pd.isnull(x) else x)
df['din'] = pd.to_datetime(df.din)
df['year'] = pd.to_datetime(df.year, format='%Y')

df['clauseyear_dum'] = df.apply(lambda x: int(1) if ( (x['year'] >= x['dif'] ) & (x['year'] <= x['din']) ) else int(0), axis=1)


"""
get combinations of countries, explode
"""

parties = pd.Series(pd.unique(df.parties))
parties_perm = parties.apply(lambda x: list(combinations(eval(x), 2)))
parties_keyer = pd.concat([parties, parties_perm], axis=1)
parties_keyer.columns = ['parties', 'parties_perm']

df = df.merge(parties_keyer, how='left', on='parties')

df=df.drop('parties', axis=1)


df=df.explode(column = 'parties_perm')
df=df.reset_index(drop=True)

df=df.drop(['dif', 'din'],axis=1)

df = df.groupby(['parties_perm', 'year', 'clause_id']).sum('clauseyear_dum')

df=df.reset_index(level=['parties_perm', 'year', 'clause_id'])

"""
pivot clauseyear sum column into x number of columns, where x is given by the number of clause_id s
"""

df=df.pivot(index=['year', 'parties_perm'], columns = 'clause_id', values='clauseyear_dum' )

df=df.fillna(int(0))

integered_col = df[df.select_dtypes([np.number]).columns].astype('int')

df = integered_col

df=df.reset_index(level=['year', 'parties_perm'])
del integered_col

####################################################################

"""
now, we bring in the BACI HS92 trade data. for bigmodel, try with just year-level, all-export level
"""


"""
import os

trade_data_folder = 'baci_hs92'

lofiles = os.listdir(trade_data_folder)


tradedata = pd.DataFrame()
for file in lofiles:
    bacidf = pd.read_csv(trade_data_folder + '/' + file)

    bacidf=bacidf.groupby(['i','j','t']).sum('v')
    bacidf=bacidf.drop('k',axis=1).reset_index(level=['i','j','t'])

    tradedata=pd.concat([tradedata, bacidf],axis=0)

"""
#join 
"""



baci_country_keyer = pd.read_csv('baci_hs92 meta/country_codes_V202201.csv', encoding = "ISO-8859-1")


# merge for exporter (i)
tradedata = tradedata.merge(baci_country_keyer[['country_code', 'iso_3digit_alpha']], how='left', left_on='i', right_on='country_code')

tradedata=tradedata.drop(['i', 'country_code'],axis=1).rename({'iso_3digit_alpha':'exporter'},axis=1)

# merge for importer (j)
tradedata = tradedata.merge(baci_country_keyer[['country_code', 'iso_3digit_alpha']], how='left', left_on='j', right_on='country_code')

tradedata=tradedata.drop(['j', 'country_code'],axis=1).rename({'iso_3digit_alpha':'importer'},axis=1)

tradedata=tradedata.rename({'t':'year', 'v':'export_value'},axis=1)

tradedata['parties_perm'] = tradedata.apply(lambda x: {x['exporter'], x['importer']},axis=1)
tradedata['year']=tradedata.year.astype('str')

"""

#########################################################################
#continue here

df['parties_perm'] = df.parties_perm.apply(lambda x: str(set(x)))
df['year'] = df.year.astype('str').apply(lambda x: int(x[0:4]))
tradedata = pd.read_csv('core_excels/tradeall.csv')

finaldf = tradedata.merge(df, how='left', on=['year', 'parties_perm'])

finaldf = finaldf.fillna(int(0))


# convert to integers to save space

integered_col = finaldf[finaldf.select_dtypes([np.number]).columns].drop('export_value',axis=1).astype('int')

exportdf = pd.concat([finaldf[['year', 'export_value', 'exporter','importer', 'parties_perm']], integered_col],axis=1)

newnames = ["x_" + str(item) for item in exportdf.columns[5:].to_list()]


orignames = exportdf.columns[5:].to_list()
orignames = [str(name) for name in orignames]

"""
mapper = {}
for idx, name in enumerate(orignames):
    mapper.update({int(name): newnames[idx]})

exportdf=exportdf.rename(mapper,axis=1)
exportdf['year']=exportdf.year.astype('int')
"""

# export and use R :D

exportdf.to_csv('analyze_exportdf.csv')


##########################################
"""
export to R to use fixest
"""



##########################################
" check baci duplicates"

tradedata

df2 = tradedata[tradedata[['exporter', 'importer', 'year']].duplicated() == True]
