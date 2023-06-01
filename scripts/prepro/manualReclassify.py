import pandas as pd

df = pd.read_csv("core_excels/totaPlus_MandatoryPredictHF.csv")
trimmer = pd.read_csv("core_excels/fta_trimmer.csv")
trimmer = trimmer.ffill(axis=1)

trydf=df.merge(trimmer, how='left', on = ['chapter_cat', 'article_cat'] )

trydf = trydf[trydf.notnull()['reassigned_artcat']]

tig_dec = pd.read_csv("core_excels/deconstruct_TIG.csv")

match = tig_dec[['chapter_cat', 'article_cat', 'in_new_chapter', 'reassigned_article_cat']]

tig_matched = match[match['in_new_chapter'].notnull()]


tig_matched = tig_matched.merge(df, how='left', on=['chapter_cat', 'article_cat'])

tig_matched = tig_matched.drop(['chapter_cat', 'article_cat'],axis=1).rename({'in_new_chapter':'chapter_cat','reassigned_article_cat':'article_cat'},axis=1)

trydf = trydf.drop('article_cat', axis=1).rename({'reassigned_artcat':'article_cat'},axis=1)

newdf = pd.concat([trydf, tig_matched],axis=0)

nomatch = tig_dec[['chapter_cat', 'article_cat', 'whole_chapter_move']]

nomatch = nomatch[nomatch.whole_chapter_move.notnull()]

nomatch = nomatch.merge(df, how='left', on=['chapter_cat', 'article_cat'])

nomatch = nomatch.drop(['chapter_cat', 'article_cat'],axis=1).rename({'whole_chapter_move':'chapter_cat'}, axis=1)

chaps = pd.unique(nomatch.chapter_cat)


import sys
sys.path.append("scripts/model/")
from cleanandTransform import cleanandTransform
from sklearn.metrics.pairwise import cosine_similarity as cs
from numpy import argmax

cat = cleanandTransform(filters = [], transformer_package='all-MiniLM-L12-v2')
cat.set_filters(['to_lower', 'remove_url','remove_special_character', 'normalize_unicode', 'remove_whitespace'])
lo_assigned_artcat = []
for chap in chaps:
    subnm = nomatch[nomatch['chapter_cat'] == chap]
    subnm=subnm.reset_index(drop=True)

    subdf = df[df['chapter_cat'] == chap]
    subdf=subdf.reset_index(drop=True)

    cat.init_text(subdf.text.astype('str'))
    cat.process_text()
    cat.transform_text()

    labeled_chap_embeds = cat.current_embedding

    cat.init_text(subnm.text.astype('str'))
    cat.process_text()
    cat.transform_text()

    unlabeled_chap_embeds = cat.current_embedding

    
    for question_mark in unlabeled_chap_embeds:
        rov = [cs([question_mark], [sample])[0][0] for sample in labeled_chap_embeds]

        assigned_artcat = subdf['article_cat'][argmax(rov)]

        lo_assigned_artcat.append(assigned_artcat)

nomatch['article_cat'] = lo_assigned_artcat

nomatch = nomatch.merge(trimmer, how='left', on=['chapter_cat', 'article_cat'])

nomatch= nomatch[nomatch.reassigned_artcat.notnull()]

nomatch = nomatch.drop('article_cat',axis=1).rename({'reassigned_artcat':'article_cat'},axis=1)

newdf=pd.concat([newdf, nomatch],axis=0)

newdf = newdf[newdf['chapter_cat'] != 'trade in goods']

newdf = newdf.reset_index(drop=True)

# redo within-article semantic clustering

newdf['clause_id'] = newdf.apply(lambda x: x['chapter_cat'] + "_" + x['article_cat'],axis=1)

newdf=newdf.drop('clauseMeaning',axis=1)

from sklearn.cluster import AffinityPropagation
semclusterModel = AffinityPropagation()
from tqdm import tqdm

smallerdfpool=pd.DataFrame()
for clausetype in tqdm(pd.unique(newdf.clause_id)):
    reltext = newdf.text[newdf['clause_id'] == clausetype].astype('str')

    smallerdf = newdf[newdf['clause_id'] == clausetype]

    cat.init_text(reltext)
    cat.process_text()
    cat.transform_text()

    clausemeaninglabels = semclusterModel.fit_predict(cat.current_embedding)

    smallerdf['clauseMeaning'] = clausemeaninglabels
    smallerdfpool = pd.concat([smallerdfpool, smallerdf],axis=0)

smallerdfpool.to_csv('core_excels/totaPlus_afterManualReclass.csv')