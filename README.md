# AFCC - Automated Feedback Constructiveness Classifier

Project aims:
- Show the possibility to classify texts based on their constructiveness
- Compare different models (state-of-the-art and outdated), different sets of features, different word embeddings
- Show that it is possible to use constructiveness as an alternative to inter-user (biased) helpfulness votes

## Table of contents:
* [Data collection](#data-collection)
  * [Training data](#training-data)
  * [Test data](#test-data)
* [Data annotation](#data-annotation)
* [Machine learning models](#machine-learning-models)
  * [Non-neural models](#non-neural-models)
  * [Neural network models](#neural-network-models)
* [Result collection and analysis](#result-collection-and-analysis)
* [Conclusion and future work](#conclusion-and-future-work)

## Data collection

### Training data

The file reviewScraper.py scrapes [IMDb](imdb.com) user reviews for movies, series and games.
We use [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) to crawl the data.

Total number of pages scraped: **1257**

Total number of reviews retrieved: **31104** 

### Test data

The test data sets will be collected later on, from different sources.

**Aim**: Test efficiency on both in-domain and out-of-domain data

## Data Annotation

TODO:
- Define a constructivenes scale (4 classes to be able to compare with previous binary work)
- Survey for annotation agreement
  - Google form with 20 reviews
  - Explain in detail each possible annotation
  - Included a question on English speaking level (A/B/C/native)
- Annotate at least 10000 reviews

## Machine learning models

TODO: everything

### Non-neural models

TODO: everything

- Naive Bayes
- Support Vector Machine

### Neural network models

TODO: everything

- Deep Convolutional network
- Reccurent (LSTM/GRU) w/ attention mechanism
- Deep Feed Forward
- 

## Result collection and analysis

TODO: everything

