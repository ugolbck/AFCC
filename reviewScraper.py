from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
from time import sleep

url = "https://www.imdb.com/search/title/?release_date=2017-01-01,2019-12-31&num_votes=500,&countries=us&sort=num_votes,desc&count=5&start=1&ref_=adv_nxt"

resp = requests.get(url)
soup = BeautifulSoup(resp.text, 'html.parser')

titles = set()
pattern = re.compile("(\/(title)\/(tt)[0-9]+\/)")

for link in soup.find_all('a'):
    # print(link.attrs)
    links = link.get('href')
    if links is not None and re.match(pattern, links) and 'vote' not in links and 'plotsummary' not in links:
        titles.add(links)
    # sleep(0.01)

print(titles)
print(len(titles))
for i in titles:
    movie_url = "https://www.imdb.com" + i + "reviews?sort=totalVotes&dir=desc&ratingFilter=0"
    print(movie_url)