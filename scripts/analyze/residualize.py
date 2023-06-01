import pandas as pd

df=pd.read_csv('analyze_exportdf.csv')


to_dummy = df[['year', 'exporter', 'importer']]

to_dummy['exporter_importer'] = to_dummy.apply(lambda x: x['exporter'] + "_" + x['importer'],axis=1)

to_dummy['exporter_year'] = to_dummy.apply(lambda x: x['exporter'] + "_" + str(x['year']),axis=1)

to_dummy['importer_year'] = to_dummy.apply(lambda x: x['importer'] + "_" + str(x['year']),axis=1)

to_dummy = to_dummy[['exporter_importer', 'exporter_year', 'importer_year']]

import pyhdfe
import numpy as np
from itertools import chain

ids = np.array(to_dummy)

algorithm = pyhdfe.create(ids)

list_of_interest = ['competition policy', 'trade facilitation', 'intellectual property rights', 'rules of origin', 'sanitary and phytosanitary measures', 'technical barriers to trade', 'trade in goods', 'e-commerce', 'investment and capital flows','movement of natural persons', 'public procurement', 'trade remedies'] #based on subregressions

filter_col = [[col for col in df.columns if col.startswith(interest)] for interest in list_of_interest]

filter_col = list(chain(*filter_col))

variables = pd.concat([df.export_value, df[filter_col]], axis=1)


residualized = algorithm.residualize(variables)

residualized = pd.DataFrame(residualized)

residualized.columns = variables.columns

import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Poisson

endog = residualized.export_value
exog = residualized.drop('export_value', axis=1)

model = Poisson(endog = endog, exog = exog)

results = model.fit()

results.summary()

"""
dead-end: if we insist on using a poisson model, then we can't have negative dependent variable values. that inevitably happens when we residualized y.
"""