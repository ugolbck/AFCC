import pandas as pd
import csv
import gzip
import json

def openShuffleTruncate(in_file, amount):
    # Open tsv file, we drop line with bugged separators
    reviews = pd.read_csv(in_file, sep='\t', header=0, error_bad_lines=False)
    # Shuffle the dataframe
    reviews = reviews.sample(frac=1).reset_index(drop=True)
    # Only keep the top <amount> rows
    reviews = reviews.truncate(after=amount)

    return reviews

def resizeSave(reviews, to_file):
    # List of fields to remove drom data files
    to_remove = ['marketplace', 'customer_id', 'review_id', 'product_id',
                'product_parent', 'product_title', 'product_category',
                'vine', 'verified_purchase', 'review_date']

    # Removing unneeded fields
    reviews = reviews.drop(to_remove, axis=1)
    print(reviews.head())

    # Append (or create) dataframe to csv file
    reviews.to_csv(to_file, mode="a+", header=False)

###########################################################

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)

def getDF(path, amount, to_file):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    reviews = pd.DataFrame.from_dict(df, orient='index')
  
    # dropping ALL duplicte values 
    reviews.drop_duplicates(subset =["reviewText"], 
                     keep = False, inplace = True) 
    
    # Shuffle the dataframe
    reviews = reviews.sample(frac=1).reset_index(drop=True)
    # Only keep the top <amount> rows
    reviews = reviews.truncate(after=amount-1)

    print(reviews.shape[0])
    
    # Save to csv
    reviews.to_csv(to_file, mode="a+", header=False)

if __name__ == "__main__":

    path_full_file = "../new_amazon_reviews.csv"
    path_new_category = "../Sports_and_Outdoors_5.json.gz"

    # resizeSave(openShuffleTruncate(path_new_category, 1500), path_full_file)
    getDF(path_new_category, 5000, path_full_file)











