import pandas as pd
import csv


path = "../amazon_reviews_multilingual_US_v1_00.tsv"

def openShuffleSave(filename):
    reviews = pd.read_csv(filename, sep='\t', header=0, error_bad_lines=False)
    reviews.head()

    # df = df.sample(frac=1).reset_index(drop=True)

openShuffleSave(path)