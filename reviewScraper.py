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
    

def scraper(movie_links):
    """ Scapes IMDb website to retrieve user reviews sorted by total number of votes
    
    param movie_links: set or list of movies/series IDs to pass to the review page's url
    return : list of lists containing review/helpful votes/total number of votes """

    rows = []
    for i in tqdm(movie_links):
        # Searching for top 25 voted reviews
        movie_url = "https://www.imdb.com/title/tt" + i + "/reviews?sort=totalVotes&dir=desc&ratingFilter=0"
        soup = BeautifulSoup(requests.get(movie_url).text, 'html.parser')

        # Creating review entry
        for review, score in zip(soup.find_all("div", class_="text show-more__control"), soup.find_all("div", class_="actions text-muted")):
            votes = [int(s.replace(',', '')) for s in score.text.split() if s.replace(',', '').isdigit()]
            rows.append([review.text, votes[0], votes[1]])
    
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
    movies = linkCollecter(250, 1251)
    print(movies)
    lines = scraper(movies)
    toCSV(lines, "justincase.csv")

