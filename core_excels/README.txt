totaPlus

basis
- branched off of Texts of Trade Agreement (TOTA) https://github.com/mappingtreaties/tota
- TOTA is a corpus of main-body preferential trade agreement (PTA) texts signed between countries
- it is meant to track trade agreements not signed on a purely multilateral level (ie. WTO/GATT)
- TOTA is updated up till 2017
- totaPlus (presently, 4 Jul 2022) updates TOTA with the remainder of PTAs signed by Singapore up till 2022

- also takes a similar philosophy to two reference studies: World Bank's Deep Trade Agreements (WBDTA) <https://datatopics.worldbank.org/dta/table.html> and Design of Trade Agreements (DESTA) <https://www.designoftradeagreements.org/>
- that is, each PTA can be regarded to have certain distinctive characteristics that significantly impact trade flows, and that these characteristics can be meaningfully distinguished by close reading of clauses
- in so doing, WBDTA and DESTA have to create both CHAPTER-LEVEL (eg. competition policy, e-commerce, sanitary and phytosanitary measures) and ARTICLE-LEVEL (eg. Objectives, Definitions, Technical Cooperation; all these being a subset of a given chapter) categories
- however, WBDTA and DESTA's approach has been to inspect, then manually create categories to, once again, manually bin articles in

what we do with totaPlus
- we use machine learning to perform semi-unsupervised classification of chapters and articles, thus refining prior results from WBDTA and DESTA
- we bring TOTA up to date till 2022

<aspirational>

- bring TOTA up to date till 2022 for other countries
- form a nice website to be a reference library for trade negotiators












// technical details
- all PTAs up till 2017: from TOTA
- Singapore PTAs 2017-2022: scraped from ESG website (scripts/prepro/scrapers)
- unite TOTA and ESG datasets (scripts/prepro/uniteTOTA_SG.py)
- note: I break down by CLAUSE-LEVEL
chapter > article > clause

> cleanandTransform.py
- I use cleanandTransform() as an object to do text preprocessing and sentence transformation under the same hood

> isSubstantive.py, trfModel/substantiveFineTuning
- I manually label clauses by whether they have substantive content (according to the legal definition, that is, whether any rights or obligations are established, no matter how weak or unenforceable they are)
- Then I do setfit (fine-tune pre-trained mini-LM sentence model > train classification SVM model) <https://towardsdatascience.com/sentence-transformer-fine-tuning-setfit-outperforms-gpt-3-on-few-shot-text-classification-while-d9a3788f0b4e>
- And use this to predict whether clauses have substantive content, or whether they are simply fluff or procedural

> standardizeChapters.py, trfModel/chapterFineTuning
- Then, I use a previously manually labelled chapter multiclass classification to setfit 
- Use this to predict whether a certain chapter title should be within a certain chapter category (eg. COMPETITION in 'competition policy')
- chapter_cat

> standardizeClauseTypes_ArticleEmbed.py
- turn raw clause texts into all mp net sentence embeddings
- WITHIN a chapter category cluster, use AFFINITY PROPAGATION to cluster clause texts
- has good results, but I find it a bit unaggressive (eg. 'definition' and 'definitions and scope' are 2 different clusters)
- clauseSem, clauseSem_no

> standardizeClauseTypes_TextEmbed.py
- Turn article TITLES into all mp net sentence embeddings
- WITHIN a chapter category cluster, use AFFINITY PROPAGATION to cluster article titles
- article_cat, article_cat_no

