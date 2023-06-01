import pandas as pd
from tqdm import tqdm

df = pd.read_csv('other_source/tota_Mandated.csv')

df = df[df['obl'].str.split().str.len().gt(5)]  # a hard and fast to avoid random uninformative obligations


fulldf  = pd.DataFrame()

for now_chapter in tqdm(pd.unique(df.chapter_cat)):

    subdf = df[df['chapter_cat'] == now_chapter]
    subdf = subdf[subdf['isSubstantive_predict'] == 'isSubstantive']

    from bertopic import BERTopic

    # do basic topic modeling - essentially getting ur top words in the topic and representative documents

    docs = subdf.obl
    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(docs)
    topic_model.get_topic_info()

    """
    following this, we use text2text generation (google's flan-t5) to generate short titles. they work reasonably well, although apparently gpt-3.5 is better. sadly it's rate limited, so maybe if you have another intern with a lot of time they can manually apply the labels lol
    """

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base") #1gb download beware
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

    summary = topic_model.get_topic_info()

    lot = []

    summary ['prompt'] = summary.apply(lambda x: "I have a topic that contains the following documents: " + str(x['Representative_Docs']).replace("\\","") + ". The topic is described by the following keywords: " + str(x['Representation']).replace('\\', '') + ". Based on the information above, extract a short topic label of less than 5 words.", axis=1)

    summary['topic'] = summary.apply(lambda x: tokenizer.batch_decode(model.generate(**tokenizer(x['prompt'], return_tensors="pt")), skip_special_tokens=True)[0], axis=1)

    subdf['topic_no'] = topics

    subdf = subdf.merge(summary[['Topic', 'topic']].rename({'Topic': 'topic_no'},axis=1), 'left')

    fulldf = pd.concat([fulldf, subdf], axis=0)

fulldf.to_csv('other_source/final_tota.csv')