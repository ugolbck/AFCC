# Award winner for "most disgusting code" 
# -> clean when I have time

import pandas as pd
import numpy as np
import os, time, re
import sys, html, string, json

import textstat
import nltk
from nltk.tokenize import casual_tokenize
from num2words import num2words
from pycorenlp import StanfordCoreNLP
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from src.words import contraction_mapping, discourse_markers, modals, softeners


def early_check(data, text_column, tag_column):
    assert isinstance(data, pd.DataFrame), 'Wrong type!'
    assert text_column in data.columns, "column {} was not found in DataFrame.".format(text_column)
    assert tag_column in data.columns, "column {} was not found in DataFrame.".format(tag_column)

    if 'Unnamed: 0' in data.columns:
        return data.drop('Unnamed: 0', axis=1)
    data = data.dropna(subset=[tag_column])
    data = data.reset_index(drop=True)
    return data

def tr_te_split(data, tag_column, te_size=0.1):
    assert tag_column in data.columns, "No column {} found in DataFrame.".format(tag_column)

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=te_size)
    for train_index, test_index in sss1.split(data, data[tag_column]):
        strat_train = data.loc[train_index]
        strat_test = data.loc[test_index]

    return strat_train, strat_test

def to_numeric(data, tag_column, pattern='abcd'):
    if pattern == 'abcd':
        data[tag_column] = data.loc[:, tag_column].map({'a': 0, 'b': 1, 'c': 2, 'd': 3})
        data['bin_tag'] = data.loc[:, tag_column].map({0: 0, 1: 0, 2: 1, 3: 1})
    elif pattern == 'yesno':
        data[tag_column] = data.loc[:, tag_column].map({'no': 0, 'yes': 1})
    else:
        raise ValueError("Wrong pattern entered.")

def html_url_cleaning(data, text_column):
        data[text_column] = [re.sub(r'(<.*?>)|((\[\[).*(\]\]))', '', html.unescape(x)) for x in data.loc[:, text_column]]
        data[text_column] = [re.sub(r'(\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*)', '<url>', x) for x in data.loc[:, text_column]]

def tokenize(data, text_column):
        if type(data[text_column][0]) is list:
            data[text_column] = data.loc[:, text_column].map(lambda x: casual_tokenize(' '.join(x)))
        else:
            data[text_column] = data.loc[:, text_column].map(lambda x: casual_tokenize(x))

def split_join(data, text_column):
    data[text_column] = data.loc[:, text_column].map(lambda x: ' '.join(x.split()))

def join_split(data, text_column):
    data[text_column] = data.loc[:, text_column].map(lambda x: (' '.join(x)).split())

def join_words(data, text_column):
    data[text_column] = [' '.join(x) for x in data.loc[:, text_column]]

def to_lower(data, text_column):
    data[text_column] = [[x.lower() for x in i] for i in data[text_column]]

def expand(data, text_column):
    data[text_column] = data.loc[:, text_column].map(lambda x: [contraction_mapping[i] if i in contraction_mapping else i for i in x])

def count_upper(data, text_column):
    data['num_upper'] = data.loc[:, text_column].map(lambda x: sum([1 for i in x if i.isupper() and i != 'I']))

def count_punct(data, text_column):
    data['num_punct'] = data.loc[:, text_column].map(lambda x: sum([1 for i in x if i in set(string.punctuation)]))

def count_positive(data, text_column, pos_lexicon):
    data['num_positives'] = data.loc[:, text_column].map(lambda x: sum([1 for i in x if i in pos_lexicon]))

def count_negative(data, text_column, neg_lexicon):
    data['num_negatives'] = data.loc[:, text_column].map(lambda x: sum([1 for i in x if i in neg_lexicon]))

def count_punct(data, text_column):
    data['num_punct'] = data.loc[:, text_column].map(lambda x: sum([1 for i in x if i in set(string.punctuation)]))

def count_unknown(data, text_column, word_dict):
    data['num_unk'] = data.loc[:, text_column].map(lambda x: sum([1 for i in x if i not in word_dict]))

def remove_punct(data, text_column):
    data[text_column] = [[x for x in i if x not in set(string.punctuation)] for i in data[text_column]]

def num_to_words(data, text_column):
    data[text_column] = [[num2words(x) if x.isdigit() else x for x in i] for i in data[text_column]]

def annotate(data, text_column, tagger):
    assert isinstance(tagger, StanfordCoreNLP)

    tags, lemmas = [], []
    for review in data[text_column]:
        if isinstance(review, list):
            review = ' '.join(review)
        annot = tagger.annotate(review,
            properties={
                'annotators': 'pos,lemma',
                'outputFormat': 'json',
                'timeout': 10000,
            })
        try:
            tags.append(' '.join([x['pos'] for x in annot['sentences'][0]['tokens']]))
            lemmas.append(' '.join([x['lemma'] for x in annot['sentences'][0]['tokens']]))
        except:
            tags.append('')
            lemmas.append('')
    data['text_pos'] = tags
    data['lemmas'] = lemmas

def ner_extract(data, text_column, tagger):
    assert isinstance(tagger, StanfordCoreNLP)

    entities = []
    for review in data[text_column]:
        if isinstance(review, list):
            review = ' '.join(review)
        annot_doc = tagger.annotate(review,
            properties={
                'annotators': 'ner',
                'outputFormat': 'json',
                'timeout': 10000,
            })
        try:
            entities.append(sum([1 for x in annot_doc['sentences'][0]['tokens'] if x['ner'] != "O"]))
        except:
            entities.append(0)
    data['named_entities'] = entities

def discourse_features(data, text_column):
    discourse, modal = [], []
    for sent in data[text_column]:
        cnt_discourse, cnt_modal = 0, 0
        if isinstance(sent, str):
            sent = sent.split(' ')

        for i in range(len(sent)):
            # Match individual word in lists
            if sent[i] in discourse_markers:
                cnt_discourse += 1
            if sent[i] in modals:
                cnt_modal += 1

            # Match several word in lists
            if i > 0:
                if sent[i-1] + ' ' + sent[i] in discourse_markers:
                    cnt_discourse += 1
            if i > 1:
                if sent[i-2] + ' ' + sent[i-1] + ' ' + sent[i] in discourse_markers:
                    cnt_discourse += 1

        discourse.append(cnt_discourse)
        modal.append(cnt_modal)
    data['num_discourse'] = discourse
    data['num_modals'] = modal

def sentiment(data, text_column, anal):
    data['sentiment'] = [anal.polarity_scores(x)['compound'] for x in data[text_column]]

def flesch_ease(data, text_column):
    data['read_score'] = [textstat.flesch_reading_ease(x) for x in data[text_column]]

def length_features(data, text_column):
    data['num_tokens'] = [len(x) for x in data[text_column]]
    data['num_char'] = [sum([len(x) for x in i]) for i in data[text_column]]
    data['avg_word_length'] = data['num_char'] / data['num_tokens']