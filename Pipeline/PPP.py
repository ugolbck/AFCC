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
        self.red_word_len = red_word_len
        self.to_drop = to_drop

        self.df = pd.read_csv(os.path.join(self.filepath, self.filename))
        
        assert self.col_name in self.df.columns, "The argument '{}' cannot be found in the columns of argument 'df'.".format(self.col_name)
        
        if self.corenlp:
            print("Make sure that the CoreNLP server is up and running.\n"
            "This scripts calls the localhost:9000")
            self.nlp_tagger = StanfordCoreNLP('http://localhost:' + str(self.port))


    def full_process(self):
        """ Wrapper method that preprocesses a csv file,
        extracts features and outputs 2 (stratified split)
        csv train/test files.
        """
        self.tag_manager()
        self.df = self.remove_missing(self.df)
        self.df = self.reset_ind(self.df.iloc[1:5, :])

        print("FRAME SHORTENED")
        self.html_url_cleaning()

        self.to_lower()

        if self.expansion:
            print("EXPANSION...")
            self.expand(contraction_mapping)
        
        print("PUNCT REMOVE")
        self.remove_punct()

        print("NUM TO WORDS")
        self.num_to_words()
        if self.corenlp:
            self.annotate(self.nlp_tagger)

        print("LENGTH FEATURES")
        self.length_features()

        self.join_words()

        print("READABILITY")
        self.flesch_ease()

        n_rows = str(self.getRows(self.df))
        train, test = self.tr_te_split()

        # self.output('./output', train_d=(train, n_rows+'_train.csv'), test_d=(test, n_rows+'_test.csv'))

        self.head(train)
        self.head(test)

    def head(self, frame):
        print(frame.head())

    def tag_manager(self):
        self.df = self.df.dropna(subset=['tag']).copy()
        self.df['tag'] = self.df.loc[:, 'tag'].map({'a': 0, 'b': 1, 'c': 2, 'd': 3,})
        self.df['bin_tag'] = self.df.loc[:, 'tag'].map({0: 0, 1: 0, 2: 1, 3: 1})
    
    
    def remove_missing(self, frame):
        """ Helper method """
        return frame.dropna(subset=['tag'])

    def reset_ind(self, frame):
        """ Helper method """
        return frame.reset_index(drop=True)
    
    def checkup(self, frame):
        """ Helper method
        Checks for missing value, in which
        case the corresponfing rows are removed
        and the the index reseted. Otherwise,
        index is reseted anyway.
        """
        if frame.isnull().values.any():
            print("NaN values found")
            return self.reset_ind(self.remove_missing(frame))
        return self.reset_ind(frame)

    def getRows(self, frame):
        return frame.shape[0]

    def getColumns(self, frame):
        return frame.shape[1]
    
    def html_url_cleaning(self):
        self.df[self.col_name] = [re.sub(r'(<.*?>)|((\[\[).*(\]\]))|(\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*)', '', html.unescape(x)) for x in self.df.loc[:, self.col_name]]
        self.tokenize()
    
    def tokenize(self):
        if type(self.df[self.col_name][0]) is not list:
            self.df[self.col_name] = self.df.loc[:, self.col_name].map(lambda x: casual_tokenize(x, reduce_len=self.red_word_len))
        else:
            self.df[self.col_name] = self.df.loc[:, self.col_name].map(lambda x: casual_tokenize(' '.join(x), reduce_len=self.red_word_len))
        
    def to_lower(self):
        self.df[self.col_name] = [[x.lower() for x in i] for i in self.df[self.col_name]]

    def expand(self, contrac_map):
        self.df[self.col_name] = self.df.loc[:, self.col_name].map(lambda x: [contrac_map[i] if i in contrac_map else i for i in x])
        self.tokenize()
    
    def remove_punct(self):
        self.df[self.col_name] = [[x for x in i if x not in set(string.punctuation)] for i in self.df[self.col_name]]
    
    def num_to_words(self):
        self.df[self.col_name] = [[num2words(x) if x.isdigit() else x for x in i] for i in self.df[self.col_name]]
        self.tokenize()
    
    def annotate(self, tagger):
        tags, lemmas = [], []
        for i in self.df[self.col_name]:
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

    def length_features(self):
        """ Gets number of tokens
        Works when review is a list of tokens """
        self.df['num_tokens'] = [len(x) for x in self.df.loc[:, self.col_name]]
        self.df['num_char'] = [sum([len(x) for x in i]) for i in self.df.loc[:, self.col_name]]
    
    def join_words(self):
        self.df[self.col_name] = [' '.join(x) for x in self.df.loc[:, self.col_name]]

    def tr_te_split(self, te_size=0.1):
        assert 'tag' in self.df.columns, "No column 'tag' found in DataFrame."
        self.df = self.checkup(self.df)

        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=te_size)
        for train_index, test_index in sss1.split(self.df, self.df['tag']):
            strat_train = self.df.loc[train_index]
            strat_test = self.df.loc[test_index]
        
        strat_train = self.checkup(strat_train)
        strat_test = self.checkup(strat_test)

        return strat_train, strat_test
        
    def flesch_ease(self):
        """ Compute readability measure for each review.
        """
        self.df['read_score'] = [textstat.flesch_reading_ease(x) for x in self.df.loc[:, self.col_name]]
    
    def cause_markers(self):
        """ Method that adds a 'cause' discourse feature.
        Spots if markers of cause like 'because' are
        present in the review.
        """
        # as a result of
        # because
        # because of
        # due to
        # thanks to
        pass

    
    def output(self, outdir, **out_data):
        for val in out_data.values():
            val[0].to_csv(os.path.join(outdir, val[1]))


class DataHolder:
    """ Holds a pandas.DataFrame object an all
    necessary methods to perform preprocessing
    and feature engineering.
    """
    # todo:
    # add sentiment and sentiment lexicon
    # clean PPPipeline and add some wrappers
    # try on real data

    def __init__(self, filepath, filename):
        self.df = pd.read_csv(os.path.join(filepath, filename))

        analyzer = SentimentIntensityAnalyzer()

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
            self.df[column] = self.df.loc[:, column].map({'a': 0, 'b': 1, 'c': 2, 'd': 3,})
            self.df['bin_tag'] = self.df.loc[:, column].map({0: 0, 1: 0, 2: 1, 3: 1})
        elif scheme == 'yesno':
            raise NotImplementedError('Feature soon to appear!')
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

