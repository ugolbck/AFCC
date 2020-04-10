import pandas as pd
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
from sklearn.model_selection import StratifiedShuffleSplit
from contraction import contraction_mapping
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

FILEDIR = './files'
if FILEDIR not in sys.path:
    sys.path.insert(1, FILEDIR)


class PPPipeline:
    """ Preprocessing pipeline. Uses a DataHolder
    object to perform series of operations on it. """

    def __init__(self, holder, corenlp=True, port=9000, expansion=True):
        
        self.holder = holder
        self.corenlp = corenlp
        self.port = port
        self.expansion = expansion
        self.analyzer = SentimentIntensityAnalyzer()
        
        if self.corenlp:
            print("Make sure that the CoreNLP server is up and running.\n"
            "This scripts listens to localhost:9000")
            self.nlp_tagger = StanfordCoreNLP('http://localhost:' + str(self.port))

    def train_test_split(self, data):
        pass

    def process_test_data(self, data):
        pass

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


class DataHolder:
    """ Holds a pandas.DataFrame object an all
    necessary methods to perform preprocessing
    and feature engineering.
    """

    def __init__(self, filepath, filename):
        self.df = pd.read_csv(os.path.join(filepath, filename))


    def head(self, frame=None):
        if frame is not None and isinstance(frame, pd.DataFrame):
            print(frame.head())
        else:
            print(self.df.head(30))
    
    def dropna(self, column:str, frame):
        frame.dropna(subset=[column], inplace=True)

    def reset_ind(self, frame):
        frame.reset_index(drop=True, inplace=True)

    def drop_reset(self, column:str, frame):
        self.dropna(column, frame)
        self.reset_ind(frame)
    
    def reset_drop(self, column:str, frame):
        self.reset_ind(frame)
        self.dropna(column, frame)
    
    def checkup(self):
        if self.df.isnull().values.any():
            return True
        return False

    def getRows(self):
        return self.df.shape[0]

    def getColumns(self):
        return self.df.shape[1]

    def to_numeric(self, column:str, scheme='abcd'):
        if scheme == 'abcd':
            self.df[column] = self.df.loc[:, column].map({'a': 0, 'b': 1, 'c': 2, 'd': 3})
            self.df['bin_tag'] = self.df.loc[:, column].map({0: 0, 1: 0, 2: 1, 3: 1})
        elif scheme == 'yesno':
            self.df[column] = self.df.loc[:, column].map({'yes': 0, 'no': 1})
        else:
            raise ValueError('Wrong scheme name!')

    def html_url_cleaning(self, column:str):
        self.df[column] = [re.sub(r'(<.*?>)|((\[\[).*(\]\]))|(\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*)', '', html.unescape(x)) for x in self.df.loc[:, column]]

    def tokenize(self, column:str):
        if type(self.df[column][0]) is list:
            self.df[column] = self.df.loc[:, column].map(lambda x: casual_tokenize(' '.join(x), reduce_len=True))
        else:
            self.df[column] = self.df.loc[:, column].map(lambda x: casual_tokenize(x, reduce_len=True))

    def split_join(self, column:str):
        self.df[column] = self.df.loc[:, column].map(lambda x: ' '.join(x.split()))

    def join_split(self, column:str):
        self.df[column] = self.df.loc[:, column].map(lambda x: (' '.join(x)).split())

    def join_words(self, column:str):
        self.df[column] = [' '.join(x) for x in self.df.loc[:, column]]

    def to_lower(self, column:str):
        self.df[column] = [[x.lower() for x in i] for i in self.df[column]]

    def expand(self, column:str):
        self.df[column] = self.df.loc[:, column].map(lambda x: [contraction_mapping[i] if i in contraction_mapping else i for i in x])

    def remove_punct(self, column:str):
        self.df[column] = [[x for x in i if x not in set(string.punctuation)] for i in self.df[column]]

    def num_to_words(self, column:str):
        self.df[column] = [[num2words(x) if x.isdigit() else x for x in i] for i in self.df[column]]

    def annotate(self, column:str, tagger):
        tags, lemmas = [], []
        for i in self.df[column]:
            annot = tagger.annotate(' '.join(i),
                properties={
                    'annotators': 'pos,lemma',
                    'outputFormat': 'json',
                    'timeout': 10000,
                })
            
            tags.append(' '.join([x['pos'] for x in annot['sentences'][0]['tokens']]))
            lemmas.append(' '.join([x['lemma'] for x in annot['sentences'][0]['tokens']]))

        self.df['text_pos'] = tags
        self.df['lemmas'] = lemmas

    def sentiment(self, column:str, anal):
        self.df['sentiment'] = [anal.polarity_scores(x)['compound'] for x in self.df[column]]

    def flesch_ease(self, column):
        self.df['read_score'] = [textstat.flesch_reading_ease(x) for x in self.df.loc[:, column]]

    def length_features(self, column:str, char_level=True):
        self.df['num_tokens'] = [len(x) for x in self.df.loc[:, column]]
        if char_level:
            self.df['num_char'] = [sum([len(x) for x in i]) for i in self.df.loc[:, column]]

    def tr_te_split(self, base='tag', te_size=0.1):
        assert base in self.df.columns, "No column {} found in DataFrame.".format(base)

        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=te_size)
        for train_index, test_index in sss1.split(self.df, self.df[base]):
            strat_train = self.df.loc[train_index]
            strat_test = self.df.loc[test_index]

        return strat_train, strat_test

    def output(self, outdir, **out_data):
        for val in out_data.values():
            val[0].to_csv(os.path.join(outdir, val[1]))

if __name__ == "__main__":
    
    my_frame = DataHolder('../data/raw_data', '2830_reviews.csv')

