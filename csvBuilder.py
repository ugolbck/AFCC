import pandas as pd
import csv

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


if __name__ == "__main__":

    path_full_file = "../amazon_reviews.csv"
    path_new_category = "../amazon_reviews_us_Electronics_v1_00.tsv"

    resizeSave(openShuffleTruncate(path_new_category, 1500), path_full_file)

