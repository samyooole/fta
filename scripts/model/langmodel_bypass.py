import pandas as pd

right = pd.read_csv('scripts/model/newtota_preSubstantiate.csv')
predictor = pd.read_csv('core_excels/totaPlus.csv')
sub = predictor[['chapter', 'article', 'clauseno', 'fta_name', 'isSubstantive_predict', 'chapter_cat', 'article_cat', 'article_cat_no', 'clauseSem', 'clauseSem_no']]

right = right.merge(sub, how='left', on=['chapter', 'article', 'clauseno', 'fta_name'])

right=right.drop('Unnamed: 0', axis=1)

right.to_csv('scripts/model/totaPlus_bypass.csv')