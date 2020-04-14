# TODO:
# Bug in _remove_punct() only for SOCC corpus. Encoding? weird punctuation?

import pandas as pd
import numpy as np
import html
import re
import string
import os
import time
import sys
import json
import nltk
import textstat
from nltk.tokenize import casual_tokenize
from pycorenlp import StanfordCoreNLP
from num2words import num2words
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from .contraction import contraction_mapping
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

""" Globals """

# Stanford NLP pipeline access
nlp_tagger = StanfordCoreNLP('http://localhost:9000')

# VADER sentiment analysis tool
analyzer = SentimentIntensityAnalyzer()

# Expand word contractions or not
EXPANSION = True

FILEDIR = './files'
if FILEDIR not in sys.path:
    sys.path.insert(1, FILEDIR)


""" Wrapper functions """

def file_split(data, test_size=0.2, text_column='text_review', tag_column='tag', out_dir=None):
    data = data[data[tag_column] != 'n']
    
    data = _early_check(data, text_column, tag_column)

    train, test = _tr_te_split(data, tag_column, te_size=test_size)
    
    train = _early_check(train, text_column, tag_column)
    test = _early_check(test, text_column, tag_column)

    if out_dir:
        train.to_csv(os.path.join(out_dir, train.shape[0]+'_train.csv'))
        test.to_csv(os.path.join(out_dir, test.shape[0]+'_test.csv'))

    return train, test


def preprocess_train(data, text_column='text_review', tag_column='tag', pattern='abcd', out_dir=None):

    data = _early_check(data, text_column, tag_column)
    data = data.loc[:, [text_column, tag_column]]
    _to_numeric(data, tag_column, pattern)
    _html_url_cleaning(data, text_column)
    _flesch_ease(data, text_column)
    _sentiment(data, text_column, analyzer)
    _tokenize(data, text_column)
    _to_lower(data, text_column)
    _expand(data, text_column)
    _join_split(data, text_column)
    _num_to_words(data, text_column)
    _join_split(data, text_column)
    _length_features(data, text_column, char_level=True)
    _join_words(data, text_column)
    data = _early_check(data, text_column, tag_column)

    _annotate(data, text_column, nlp_tagger)
    
    train, val = _tr_te_split(data, tag_column)
    train = _early_check(train, text_column, tag_column)
    val = _early_check(val, text_column, tag_column)

    if out_dir:
        train.to_csv(os.path.join(out_dir, train.shape[0]+'_train.csv'))
        val.to_csv(os.path.join(out_dir, val.shape[0]+'_val.csv'))
    
    return train, val


def preprocess_test(data, text_column='text_review', tag_column='tag', pattern='abcd'):

    data = _early_check(data, text_column, tag_column)
    data = data.loc[:, [text_column, tag_column]]
    _to_numeric(data, tag_column, pattern)
    _html_url_cleaning(data, text_column)
    _flesch_ease(data, text_column)
    _sentiment(data, text_column, analyzer)
    _tokenize(data, text_column)
    _to_lower(data, text_column)
    _expand(data, text_column)
    _join_split(data, text_column)
    _num_to_words(data, text_column)
    _join_split(data, text_column)
    _length_features(data, text_column, char_level=True)
    _join_words(data, text_column)
    data = _early_check(data, text_column, tag_column)
    _annotate(data, text_column, nlp_tagger)
    
    return data

    
""" Helper functions """

def _early_check(data, text_column, tag_column):
    assert isinstance(data, pd.DataFrame), 'Wrong type!'
    assert text_column in data.columns, "column {} was not found in DataFrame.".format(text_column)
    assert tag_column in data.columns, "column {} was not found in DataFrame.".format(tag_column)

    if 'Unnamed: 0' in data.columns:
        return data.drop('Unnamed: 0', axis=1)
    data = data.dropna(subset=[tag_column])
    data = data.reset_index(drop=True)
    return data

def _tr_te_split(data, tag_column, te_size=0.1):
    assert tag_column in data.columns, "No column {} found in DataFrame.".format(tag_column)

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=te_size)
    for train_index, test_index in sss1.split(data, data[tag_column]):
        strat_train = data.loc[train_index]
        strat_test = data.loc[test_index]

    return strat_train, strat_test

def _to_numeric(data, tag_column, pattern='abcd'):
    if pattern == 'abcd':
        data[tag_column] = data.loc[:, tag_column].map({'a': 0, 'b': 1, 'c': 2, 'd': 3})
        data['bin_tag'] = data.loc[:, tag_column].map({0: 0, 1: 0, 2: 1, 3: 1})
    elif pattern == 'yesno':
        data[tag_column] = data.loc[:, tag_column].map({'no': 0, 'yes': 1})
    else:
        raise ValueError("Wrong pattern entered.")

def _html_url_cleaning(data, text_column):
        data[text_column] = [re.sub(r'(<.*?>)|((\[\[).*(\]\]))|(\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*)', '', html.unescape(x)) for x in data.loc[:, text_column]]

def _tokenize(data, text_column):
        if type(data[text_column][0]) is list:
            data[text_column] = data.loc[:, text_column].map(lambda x: casual_tokenize(' '.join(x)))
        else:
            data[text_column] = data.loc[:, text_column].map(lambda x: casual_tokenize(x))

def _split_join(data, text_column):
    data[text_column] = data.loc[:, text_column].map(lambda x: ' '.join(x.split()))

def _join_split(data, text_column):
    data[text_column] = data.loc[:, text_column].map(lambda x: (' '.join(x)).split())

def _join_words(data, text_column):
    data[text_column] = [' '.join(x) for x in data.loc[:, text_column]]

def _to_lower(data, text_column):
    data[text_column] = [[x.lower() for x in i] for i in data[text_column]]

def _expand(data, text_column):
    data[text_column] = data.loc[:, text_column].map(lambda x: [contraction_mapping[i] if i in contraction_mapping else i for i in x])

def _remove_punct(data, text_column):
    data[text_column] = [[x for x in i if x not in set(string.punctuation)] for i in data[text_column]]

def _num_to_words(data, text_column):
    data[text_column] = [[num2words(x) if x.isdigit() else x for x in i] for i in data[text_column]]

def _annotate(data, text_column, tagger):
    assert isinstance(tagger, StanfordCoreNLP)
    tags, lemmas = [], []
    for i in data[text_column]:
        if isinstance(i, list):
            i = ' '.join(i)
        annot = tagger.annotate(i,
            properties={
                'annotators': 'pos,lemma',
                'outputFormat': 'json',
                'timeout': 10000,
            })
        try:
            tags.append(' '.join([x['pos'] for x in annot['sentences'][0]['tokens']]))
            lemmas.append(' '.join([x['lemma'] for x in annot['sentences'][0]['tokens']]))
        except:
            print('it failed')
            print(i)
    
    data['text_pos'] = tags
    data['lemmas'] = lemmas
    

def _sentiment(data, text_column, anal):
    data['sentiment'] = [anal.polarity_scores(x)['compound'] for x in data[text_column]]

def _flesch_ease(data, text_column):
    data['read_score'] = [textstat.flesch_reading_ease(x) for x in data.loc[:, text_column]]

def _length_features(data, text_column, char_level=True):
    data['num_tokens'] = [len(x) for x in data.loc[:, text_column]]
    if char_level:
        data['num_char'] = [sum([len(x) for x in i]) for i in data.loc[:, text_column]]



class PPPipeline:   
    def process_train_data(self, data):
        self.dropna('tag', self.df)
        self.to_numeric('tag', scheme='abcd')
        if self.checkup():
            self.drop_reset('tag', self.df)
        self.html_url_cleaning('text_review')
        self.flesch_ease('text_review')
        self.sentiment('text_review', anal=analyzer)
        self.head()
        self.tokenize('text_review')
        self.to_lower('text_review')
        self.expand('text_review')
        self.join_split('text_review')
        self.remove_punct('text_review')
        self.num_to_words('text_review')
        self.join_split('text_review')
        self.length_features('text_review')
        self.join_words('text_review')
        if self.checkup():
            self.drop_reset('tag', self.df)
        else:
            self.reset_ind(self.df)
        
        train, test = self.tr_te_split(base='tag')
        self.drop_reset('tag', train)
        self.drop_reset('tag', test)

        # self.output('./output', train_d=(train, n_rows+'_train.csv'), test_d=(test, n_rows+'_test.csv'))


if __name__ == "__main__":
    
    data = pd.read_csv('../data/data/SOCC.csv')
    test = preprocess_test(data, text_column='comment_text', tag_column='is_constructive', pattern='yesno')
    # test_annot(data)