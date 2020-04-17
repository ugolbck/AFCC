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
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from Pipeline.words import contraction_mapping, discourse
from Pipeline.utils import *


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
    print("Splitting files...")

    data = data[data[tag_column] != 'n']
    
    data = _early_check(data, text_column, tag_column)

    train, test = _tr_te_split(data, tag_column, te_size=test_size)
    
    train = _early_check(train, text_column, tag_column)
    test = _early_check(test, text_column, tag_column)

    if out_dir:
        train.to_csv(os.path.join(out_dir, train.shape[0]+'_train.csv'))
        test.to_csv(os.path.join(out_dir, test.shape[0]+'_test.csv'))
    print("Splitting finished.")

    return train, test


def preprocess_train(data, text_column='text_review', tag_column='tag', pattern='abcd', split_val=True, val_size=0.1, out_dir=None):
    assert isinstance(split_val, bool), '`split_val` must be a Boolean.'

    print("Preprocessing of the training data...")
    t1 = time.time()

    data = _early_check(data, text_column, tag_column)
    data = data.loc[:, [text_column, tag_column]]
    if pattern:
        _to_numeric(data, tag_column, pattern)
    _html_url_cleaning(data, text_column)
    _flesch_ease(data, text_column)
    _sentiment(data, text_column, analyzer)
    _tokenize(data, text_column)
    _count_upper(data, text_column)
    _to_lower(data, text_column)
    _expand(data, text_column)
    _join_split(data, text_column)
    _remove_punct(data, text_column)
    _num_to_words(data, text_column)
    _join_split(data, text_column)
    _discourse(data, text_column)
    _length_features(data, text_column, char_level=True)
    _join_words(data, text_column)
    data = _early_check(data, text_column, tag_column)

    _annotate(data, text_column, nlp_tagger)
    
    if split_val:
        train, val = _tr_te_split(data, tag_column, val_size)
        train = _early_check(train, text_column, tag_column)
        val = _early_check(val, text_column, tag_column)

        if out_dir:
            train.to_csv(os.path.join(out_dir, train.shape[0]+'_train.csv'))
            val.to_csv(os.path.join(out_dir, val.shape[0]+'_val.csv'))
            print("Files saved to {}.".format(out_dir))
        
        print("Preprocessing finished in {} seconds.".format(time.time() - t1))
        return train, val
    else:
        train = _early_check(train, text_column, tag_column)
        if out_dir:
            train.to_csv(os.path.join(out_dir, train.shape[0]+'_train.csv'))
            print("File saved to {}.".format(out_dir))

        print("Preprocessing finished in {} seconds.".format(time.time() - t1))
        return train


def preprocess_test(data, text_column='text_review', tag_column='tag', pattern='abcd'):

    print("Preprocessing of the test data...")
    t1 = time.time()

    data = _early_check(data, text_column, tag_column)
    data = data.loc[:, [text_column, tag_column]]
    if pattern:
        _to_numeric(data, tag_column, pattern)
    _html_url_cleaning(data, text_column)
    _flesch_ease(data, text_column)
    _sentiment(data, text_column, analyzer)
    _tokenize(data, text_column)
    _count_upper(data, text_column)
    _to_lower(data, text_column)
    _expand(data, text_column)
    _join_split(data, text_column)
    _remove_punct(data, text_column)
    _num_to_words(data, text_column)
    _join_split(data, text_column)
    _discourse(data, text_column)
    _length_features(data, text_column, char_level=True)
    _join_words(data, text_column)
    data = _early_check(data, text_column, tag_column)
    _annotate(data, text_column, nlp_tagger)
    
    print("Preprocessing finished in {} seconds.".format(time.time() - t1))

    return data
