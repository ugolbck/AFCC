from pycorenlp import StanfordCoreNLP
import pandas as pd
import os
import json

nlp_wrapper = StanfordCoreNLP('http://localhost:9000')


data = {
    'text': ['hi my name is Ugo', 'hi my Name is ugo', 'The Eiffel Tower is very high'],
    'tag': [1, None, 10]
}

df = pd.DataFrame(data)


pos_tags = []
lemmas = []
ner = []

for i in df['text']:
    
    
    annot_doc = nlp_wrapper.annotate(i,
        properties={
        'annotators': 'pos,lemma,ner',
        'outputFormat': 'json',
        'timeout': 1000000,
        })
    pos_tags.append(' '.join([x['pos'] for x in annot_doc['sentences'][0]['tokens']]))
    lemmas.append(' '.join([x['lemma'] for x in annot_doc['sentences'][0]['tokens']]))
    ner.append(' '.join([x['ner'] for x in annot_doc['sentences'][0]['tokens']]))

    with open('out.json', 'a+') as outp:
        print(annot_doc)
        json.dump(annot_doc, outp)
        print('==========')

df['text_pos'] = pos_tags
df['lemmas'] = lemmas
df['ner'] = ner

print(df.head())

# df['pos'] = pos_tags
# df['lemmas'] = lemmas

# def output(frame, *filenames):
#     assert isinstance(frame, pd.DataFrame)
#     print([type(i) for i in filenames])
#     print(len(filenames))
#     # frame.to_csv(os.path.join('.', 'test.csv'))

# output(df, 'abc', 'def', 4)

# for sentence in annot_doc["sentences"]:
#     print ( " ".join([word["word"] for word in sentence["tokens"]]) + " => " \
#         + str(sentence["sentimentValue"]) + " = "+ sentence["sentiment"])
