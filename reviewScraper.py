""" 
Creator: Ugo LOOBUYCK
Date: January 2020
Project: Master Thesis: 'feedback constructiveness classification'
"""

from bs4 import BeautifulSoup
import requests
import re
import csv
import itertools
import pandas as pd
from time import sleep
from tqdm import tqdm

def linkCollecter(n_movies, n_max):
    """ Collects movie IDs and stores them in a set of strings

    param n_movies: number of movies/series per page, passed in the url
    param n_max: total number of movies/series to retrieve. if n_max is not a multiple of n_movies, the excedent won't be retrieved
    return: set of strings """

    assert n_movies != 0, "Number of movies per page can't be 0."
    assert n_max >= n_movies, "Number of total movies must be at least the same the number of movies per page."

    titles = set()
    
    for i in tqdm(range(1, n_max, n_movies)):
        # Search between 2016, with minimum 500 ratings, released in the US
        url = "https://www.imdb.com/search/title/?release_date=2016-01-01,2019-12-31&num_votes=500,&countries=us&sort=num_votes,desc&count=" + str(n_movies) + "&start=" + str(i) + "&ref_=adv_nxt"
        resp = requests.get(url)
        soup = BeautifulSoup(resp.text, 'html.parser')

        # Movie ID 
        pattern = re.compile("(\/(title)\/(tt)[0-9]+\/)")
        id_pattern = re.compile("([0-9]+)")

        for link in soup.find_all('a'):
            links = link.get('href')
            # Refining research in the different links 
            if links is not None and re.match(pattern, links) and 'vote' not in links and 'plotsummary' not in links:
                titles.add(re.findall(id_pattern, links)[0])
            # Sleep to avoid having our IP address banned from IMDb server
            sleep(0.05)
    print("Number of movies/series:", len(titles))
    return titles
    

def scraper(movie_links, limit):
    """ Scapes IMDb website to retrieve user reviews sorted by total number of votes.
    To avoid reviews without rating, we scrape the 25 reviews for each rating score [1-10].
    This also maximizes the number of retrieved items.
    
    param movie_links: set or list of movies/series IDs to pass to the review page's url
    param limit: approximative limit of entries in our data set
    return : list of lists containing review/helpful votes/total number of votes """

    rows = [['review', 'helpful', 'total_helpful', 'user_rating']]
    for i in tqdm(movie_links):
        for j in range(1, 11):
            if len(rows) >= limit:
                break
            # Searching for top 25 voted reviews for each rating score
            movie_url = "https://www.imdb.com/title/tt" + i + "/reviews?sort=totalVotes&dir=desc&ratingFilter=" + str(j)
            soup = BeautifulSoup(requests.get(movie_url).text, 'html.parser')

            rev = soup.find_all("div", class_="text show-more__control")
            sco = soup.find_all("div", class_="actions text-muted")
            rat = soup.find_all("span", class_="rating-other-user-rating")

            # We filter out the reviews listings that are missing a rating score (IMDb bug ??)
            length = len(rev)
            if any(len(lst) != length for lst in [sco, rat]):
                print("Rating missing in movie:", i, "at rating", str(j))
                pass

            # Creating review entry
            for review, score, rating in zip(rev, sco, rat):
                # Retrieve helpfulness ratings and total ratings
                votes = [int(s.replace(',', '')) for s in score.text.split() if s.replace(',', '').isdigit()]
                # [REVIEW | HELPFULNESS | TOTAL VOTES | USER RATING]
                rows.append([review.text, votes[0], votes[1], rating.span.text])
    
    print("Number of reviews:", len(rows))
    return rows


def toCSV(rows, filename):
    """ Turns a list of lists into a comma separated values file
    param rows: list of lists 
    param filename: name of the csv file """

    with open(filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(rows)


########## MAIN ##########
# Complete IMDb scraper for reviews retrieval
# of movies/series from 2016 to 2019 sorted
# by popularity. Reviews are sorted by number of
# "helpfulness" total votes in order to compare
# constructructiveness and helpfulness on IMDb.

if __name__ == "__main__":
    movies = linkCollecter(25, 125)
    print(movies)
    lines = scraper(movies, 20000)
    toCSV(lines, "reviews_and_ratings.csv")

