import pandas as pd
import numpy as np

import os
import time
import sys
from pycorenlp import StanfordCoreNLP
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from Pipeline.utils import *


""" Globals """

# Stanford NLP pipeline access
nlp_tagger = StanfordCoreNLP('http://localhost:9000')

# VADER sentiment analysis tool
analyzer = SentimentIntensityAnalyzer()

# Expand word contractions or not
EXPANSION = True

# positive/negative lexicon loading
PATH_LEXICON = '/Users/ugo/Documents/MPLT/work/Thesis/AFCC/Pipeline/files/'
with open(PATH_LEXICON+"positive-words.txt", "r") as positive, open(PATH_LEXICON+"negative-words.txt", "r", encoding="ISO-8859-1") as negative:
    posit_list = [x[:-1] for x in positive.readlines()]
    negat_list = [x[:-1] for x in negative.readlines()]

# English word lexicon
with open(PATH_LEXICON+'word_dict.txt') as word_file:
        valid_words = set(word_file.read().split())

""" Wrapper functions """

def file_split(data, test_size=0.2, text_column='text_review', tag_column='tag', out_dir=None):
    print("Splitting files...")
    
    data = early_check(data, text_column, tag_column)

    train, test = tr_te_split(data, tag_column, te_size=test_size)
    
    train = early_check(train, text_column, tag_column)
    test = early_check(test, text_column, tag_column)

    if out_dir:
        train.to_csv(os.path.join(out_dir, str(train.shape[0])+'_train.csv'))
        test.to_csv(os.path.join(out_dir, str(test.shape[0])+'_test.csv'))
    print("Splitting finished.")

    return train, test


def preprocess_train(data, text_column='text_review', tag_column='tag', pattern='abcd', split_val=True, val_size=0.1, out_dir=None):
    assert isinstance(split_val, bool), '`split_val` must be a Boolean.'

    print("Preprocessing of the training data...")
    t1 = time.time()

    data = early_check(data, text_column, tag_column)
    data = data.loc[:, [text_column, tag_column]]
    if pattern:
        to_numeric(data, tag_column, pattern)
    html_url_cleaning(data, text_column)
    flesch_ease(data, text_column)
    sentiment(data, text_column, analyzer)
    tokenize(data, text_column)
    count_upper(data, text_column)
    ner_extract(data, text_column, nlp_tagger)
    to_lower(data, text_column)
    expand(data, text_column)
    join_split(data, text_column)
    count_positive(data, text_column, posit_list)
    count_negative(data, text_column, negat_list)
    count_unknown(data, text_column, valid_words)
    remove_punct(data, text_column)
    num_to_words(data, text_column)
    join_split(data, text_column)
    discourse_features(data, text_column)
    length_features(data, text_column)
    join_words(data, text_column)
    data = early_check(data, text_column, tag_column)

    annotate(data, text_column, nlp_tagger)
    
    if split_val:
        train, val = tr_te_split(data, tag_column, val_size)
        train = early_check(train, text_column, tag_column)
        val = early_check(val, text_column, tag_column)

        if out_dir:
            train.to_csv(os.path.join(out_dir, str(train.shape[0])+'_train.csv'))
            val.to_csv(os.path.join(out_dir, str(val.shape[0])+'_val.csv'))
            print("Files saved to {}.".format(out_dir))
        
        print("Preprocessing finished in {} seconds.".format(time.time() - t1))
        return train, val
    else:
        train = early_check(train, text_column, tag_column)
        if out_dir:
            train.to_csv(os.path.join(out_dir, str(train.shape[0])+'_train.csv'))
            print("File saved to {}.".format(out_dir))

        print("Preprocessing finished in {} seconds.".format(time.time() - t1))
        return train


def preprocess_test(data, text_column='text_review', tag_column='tag', pattern='abcd'):

    print("Preprocessing of the test data...")
    t1 = time.time()

    data = early_check(data, text_column, tag_column)
    data = data.loc[:, [text_column, tag_column]]
    if pattern:
        to_numeric(data, tag_column, pattern)
    html_url_cleaning(data, text_column)
    flesch_ease(data, text_column)
    sentiment(data, text_column, analyzer)
    tokenize(data, text_column)
    count_upper(data, text_column)
    ner_extract(data, text_column, nlp_tagger)
    to_lower(data, text_column)
    expand(data, text_column)
    join_split(data, text_column)
    count_positive(data, text_column, posit_list)
    count_negative(data, text_column, negat_list)
    count_unknown(data, text_column, valid_words)
    remove_punct(data, text_column)
    num_to_words(data, text_column)
    join_split(data, text_column)
    discourse_features(data, text_column)
    length_features(data, text_column)
    join_words(data, text_column)
    data = early_check(data, text_column, tag_column)
    annotate(data, text_column, nlp_tagger)
    
    print("Preprocessing finished in {} seconds.".format(time.time() - t1))

    return data

if __name__ == "__main__":
    pass