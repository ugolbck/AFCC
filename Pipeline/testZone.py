from pycorenlp import StanfordCoreNLP
import pandas as pd
import os
from sklearn.model_selection import StratifiedShuffleSplit


nlp_wrapper = StanfordCoreNLP('http://localhost:9000')


data = {
    'text': [['ok', 'i', 'was', 'in', 'new', 'york', 'going', 'there'], ['yo'], ['i', 'bought', 'this', 'turbotax', 'and', 'it', 'was', 'such', 'a', 'waste', 'of', 'money']],
    'tag': [1, None, 10]
}

df = pd.DataFrame(data)

def checkup(frame):
        assert isinstance(frame, pd.DataFrame), 'Argument 1 must be a pandas.DataFrame object.'
        if frame.isnull().values.any():
            return frame.dropna(subset=['tag']).reset_index()
        return frame


df = checkup(df)
print(df.head())

# pos_tags = []
# lemmas = []

# for i in df['text']:
    
#     annot_doc = nlp_wrapper.annotate(' '.join(i),
#         properties={
#         'annotators': 'pos,lemma',
#         'outputFormat': 'json',
#         'timeout': 1000000,
#         })
#     pos_tags.append(' '.join([x['pos'] for x in annot_doc['sentences'][0]['tokens']]))
#     lemmas.append(' '.join([x['lemma'] for x in annot_doc['sentences'][0]['tokens']]))
#     print(type(annot_doc))
#     print(annot_doc['sentences'][0])


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