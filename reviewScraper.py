from bs4 import BeautifulSoup
import requests
import re
import csv
import pandas as pd
from time import sleep

for i in range(1, 1251, 250):
    url = "https://www.imdb.com/search/title/?release_date=2016-01-01,2019-12-31&num_votes=500,&countries=us&sort=num_votes,desc&count=3&start=1&ref_=adv_nxt"

    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')

    titles = set()
    pattern = re.compile("(\/(title)\/(tt)[0-9]+\/)")

    for link in soup.find_all('a'):
        links = link.get('href')
        if links is not None and re.match(pattern, links) and 'vote' not in links and 'plotsummary' not in links:
            titles.add(links)
        sleep(0.02)

reviews, scores = [], []
for i in titles:
    movie_url = "https://www.imdb.com" + i + "reviews?sort=totalVotes&dir=desc&ratingFilter=0"
    newsoup = BeautifulSoup(requests.get(movie_url).text, 'html.parser')

    for review, score in zip(newsoup.find_all("div", class_="text show-more__control"), newsoup.find_all("div", class_="actions text-muted")):
        reviews.append(review.text)
        scores.append([int(s.replace(',', '')) for s in score.text.split() if s.replace(',', '').isdigit()])
        # print(review.text[:40])
        # print([int(s.replace(',', '')) for s in score.text.split() if s.replace(',', '').isdigit()])

        # Warning if either a review or score is missing
        if len(reviews) != len(scores):
            print("Different # of reviews and scores!")
            print(len(reviews))
            print(len(scores))
    
# print(reviews)
print(len(reviews))
# print(scores)
print(len(scores))

rows = []
for i, j in zip(reviews, scores):
    rows.append([i, j[0], j[1]])

print(rows)