"""
Now that we have our totaPlus dataset, we wish to pull relevant data from the web to form a comprehensive trade dataset.

In particular, we aim towards a baseline trade gravity model with fixed effects.
"""

import pandas as pd
from itertools import permutations, combinations, chain
import requests
import json
import pycountry as cty
import statsmodels.formula.api as smf
from dask.dataframe import from_pandas

df = pd.read_csv("core_excels/totaPlus_bypass.csv")

# try filtering out only the substantive clauses

df=df[df['isSubstantive_predict'] == 1]


"""
select relevant columns<clause semantic classification>"""

df=df[['parties', 'dif', 'din', 'chapter_cat', 'article_cat']]
df['clause_id'] = df.groupby(['chapter_cat', 'article_cat']).ngroup()
df=df.drop(['chapter_cat', 'article_cat'],axis=1)
df=df.drop_duplicates(['parties', 'clause_id']).reset_index(drop=True)

"""
the main independent variable being: does this exporter-importer pair have at least one instance of this clause type?
"""

import numpy as np
newdf = pd.DataFrame(np.repeat(df.values, 74, axis=0))
newdf.columns = df.columns

newdf['year'] = list(range(1948,2022)) * int(len(newdf) / 74)

df=newdf
del newdf


"""
is a particular clause type active?
"""
df['din'] = df['din'].apply(lambda x: '2050' if pd.isnull(x) else x)
df['clauseyear_dum'] = df.apply(lambda x: 1 if ( (x['year'] >= int(x['dif'][0:4])) & (x['year'] <= int(x['din'][0:4])) ) else 0, axis=1)


df=df.drop(['din', 'dif'],axis=1)

"""
first, we have to explode the parties column. since our baseline model has trade from exporter i to importer j, we first take all permutations of the parties
"""

parties = pd.Series(pd.unique(df.parties))
parties_perm = parties.apply(lambda x: list(permutations(eval(x), 2)))
parties_keyer = pd.concat([parties, parties_perm], axis=1)
parties_keyer.columns = ['parties', 'parties_perm']

df = df.merge(parties_keyer, how='left', on='parties')

df=df.drop('parties', axis=1)

"""

"""

df=df.explode(column = 'parties_perm')
df=df.reset_index(drop=True)

df2 = pd.DataFrame(df.parties_perm.tolist(), index=df.index)
df2.columns = ['exporter', 'importer']

#df = df.drop('parties_perm', axis=1)
df[['exporter', 'importer']] = df2

df=df.drop_duplicates(['exporter', 'importer', 'year', 'clause_id']).reset_index(drop=True) # we only take at least one appearance of the clause as the threshold. also reduces the space

"""
pivot out clause_ids
"""
x=df.pivot(index=['year', 'parties_perm', 'exporter', 'importer'], columns = 'clause_id', values='clauseyear_dum' )

df = x
del x

df=df.fillna(0)
dx=df.reset_index()
df=dx
del dx

"""
use imf direction of trade statistics instead

(X, M)
exports from country X towards country M
"""

imf = pd.read_csv("scripts/analyze/imfdot.csv")
imf=imf[imf['Indicator Name'] == 'Goods, Value of Exports, Free on board (FOB), US Dollars'] # we only want export values

imf=imf.drop('Unnamed: 81',axis=1)

imf=imf.melt(id_vars = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code',
       'Counterpart Country Name', 'Counterpart Country Code', 'Attribute'],
       value_vars = ['1948', '1949', '1950', '1951', '1952', '1953', '1954', '1955', '1956',
       '1957', '1958', '1959', '1960', '1961', '1962', '1963', '1964', '1965',
       '1966', '1967', '1968', '1969', '1970', '1971', '1972', '1973', '1974',
       '1975', '1976', '1977', '1978', '1979', '1980', '1981', '1982', '1983',
       '1984', '1985', '1986', '1987', '1988', '1989', '1990', '1991', '1992',
       '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001',
       '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010',
       '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019',
       '2020', '2021'], var_name='year')

imf = imf[imf['Attribute'] == 'Value']

imf=imf[['Country Name', 'Counterpart Country Name', 'year', 'value']]

imf['Country Name']=imf['Country Name'].apply(lambda x: x.replace("Rep.", "Republic"))
imf['Counterpart Country Name']=imf['Counterpart Country Name'].apply(lambda x: x.replace("Rep.", "Republic"))


imf['Country Name']=imf['Country Name'].apply(lambda x: x.split(",")[0])
imf['Counterpart Country Name']=imf['Counterpart Country Name'].apply(lambda x: x.split(",")[0])


# iso standardization of country names

countrynames=pd.unique(imf['Country Name'].append(imf['Counterpart Country Name']))

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
    {'Taiwan Province of China': 
    
    sf('taiwan')[0].alpha_3}
)

ctydict.update(
    {"Lao People's Dem. Republic": 
    
    'LAO'}
)


imf['Country Name']=imf['Country Name'].apply(lambda x: ctydict[x] if x in ctydict.keys() else x)
imf['Counterpart Country Name']=imf['Counterpart Country Name'].apply(lambda x: ctydict[x]if x in ctydict.keys() else x)

imf.columns = ['exporter', 'importer', 'year', 'export value']

imf['year'] = imf.year.astype('int64')

imf = imf.drop_duplicates(['exporter', 'importer', 'year']).reset_index(drop=True)


"""
now we start forming the panel dataset.
simple fixed effects, with exporter-importer as the entity and year as the time
"""

df=df.merge(imf, how='left', on=['exporter', 'importer', 'year'])

df = df.set_index(['parties_perm', 'year'])



"""
drop obs from before 1959
"""
df = df.query("year >= 1959")

from linearmodels import PanelOLS

endog = df['export value']
exog = df.drop(['export value', 'exporter', 'importer',], axis=1).astype('int')

"""
how to handle all articles? why is it throwing an error"""


FE = PanelOLS(endog, exog, entity_effects=True, time_effects=True,check_rank=False, drop_absorbed=True)

result = FE.fit(cov_type = 'clustered',
             cluster_entity=True,
              cluster_time=True
             )

estims = pd.DataFrame(result.params)
estims['clause_id'] = estims.index






ndf = pd.read_csv("core_excels/totaPlus_bypass.csv")


"""
select relevant columns<article label classification>"""

ndf=ndf[['parties', 'dif', 'din', 'chapter_cat', 'article_cat']]
ndf['clause_id']=ndf.groupby(['chapter_cat', 'article_cat']).ngroup()

ndf=ndf[['chapter_cat', 'article_cat', 'clause_id']].drop_duplicates().reset_index(drop=True)

estims = estims.merge(ndf, how='left', on='clause_id')

estims.to_csv('femodel_dropnonSub.csv')















# create exporter-importer dummy

df=df.rename({'parties_perm': 'exporter_importer'}, axis=1)

# create exporter-year dummy

df['exporter_year'] = df['year'].astype('str') + " exp " + df['exporter']

# create importer-year dummy

df['importer_year'] = df['year'].astype('str') + " imp " + df['importer']

# how to handle NA export values? fill with 0?

df=df.fillna(0)

df=df.drop('year', axis=1)

exporter_year = pd.get_dummies(data=df['exporter_year'], drop_first=True)
importer_year =pd.get_dummies(data=df['importer_year'], drop_first=True)
exporter_importer = pd.get_dummies(data=df['exporter_importer'], drop_first=True)

df=df.drop(['exporter_year', 'importer_year','exporter_importer'], axis=1)
df=df.join(exporter_year)
del exporter_year
df=df.join(importer_year)
del importer_year
df=df.join(exporter_importer)
del exporter_importer

# try to get dummies


"""
run a fixed effects regression
"""
endog = df['export value']
exog = df.drop(['export value', 'exporter', 'importer',], axis=1)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(exog, endog)

endog = from_pandas(endog, npartitions=10)
exog = from_pandas(exog, npartitions=10)

del df

list_of_numbers = list(exog.columns[1:1307])

i=0
newlon=list_of_numbers
newlon.pop(i)

exog_subset = exog.drop(newlon, axis=1)


df=df.rename({'export value': 'export_value'}, axis=1) 
model = smf.ols(formula = 'export_value ~ C(exporter_importer) + C(exporter_year) + C(importer_year)', data=df)

model.fit(exog, endog)













"""
(M, X)
value of goods flowing into importer M from exporter X
- WTO bilat imports is only until 1996.
"""
wto = pd.read_csv("core_excels/wto_bilatImports.csv",encoding = "ISO-8859-1")

wto = wto[['Indicator', 'Reporting Economy ISO3A Code', 'Partner Economy ISO3A Code', 'Product/Sector', 'Year', 'Value', 'Unit']]

wto = wto.groupby(['Reporting Economy ISO3A Code', 'Partner Economy ISO3A Code', 'Year'], as_index=False)['Value'].sum()

wto.columns = ['importer', 'exporter', 'year', 'value']

wto[ (wto['importer'] == 'USA') & (wto['exporter'] == 'SGP')]