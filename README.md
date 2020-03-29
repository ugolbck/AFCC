# AFCC - Automated Feedback Constructiveness Classifier

Author: **Ugo Loobuyck**
Purpose: **Master's Thesis**

Project aims:
- Show the possibility to classify feedback reviews based on constructiveness:
    - Provide a valid and reliable constructiveness scale.
    - Compare different models, different sets of features and numberof features.
    - Compare our scale results with binary classification and out-of-domain data.
    

## Table of contents:
* [Constructiveness](#constructiveness)
* [Data](#data)
  * [Training data](#training-data)
  * [Test data](#test-data)
* [Data annotation](#data-annotation)
* [Features](#features)
* [Machine Learning Models](#machine-learning-models)
  * [Non-Neural](#non-neural)
  * [Neural Network](#neural-network)
* [Feature Selection](#feature-selection)
* [Results and analysis](#results-and-analysis)
* [Conclusion and Future Work](#conclusion-and-future-work)

## Constructiveness

Constructiveness for reviews:

- Respectful
- States qualities and flaws
- Turning negative into positive
- Good spelling and phrasing
- Decent length
- Objective rather than subjective
- Author has actual experience with the subject of the review

...

## Data

### Training data

??? product reviews from the official Amazon data sets were collected randomly across
the available categories (see Thesis Appendix for complete list).

### Test data

- ??? Amazon reviews
- ??? Yelp reviews
- ??? Wine reviews

## Data Annotation

## Features

We use different features for this classification task, including:

- High-level features (length...)
- Lexical features (# of postive words, ratio of uppercase words...)
- Syntactic features (POS tags...)
- Discourse features (argumentation indicators...)

## Machine Learning Models

We use scikit-learn for statistical models, SVMs and tree-based models.
We use Keras (TensorFlow backend) for neural network models.

All results are obtained on a validation set to determine the best set of features

### Non-Neural

Most promising model for:

- Multiclass:
    RandomForest (~ 62% F1) (all features)

- Binary:
    Logistic Regression (~ 78% F1) (all features)

### Neural Network

We train a RNN model with and without attention mechanism. We use GloVe word embeddings pre-trained on Twitter with 200 dimensions.

TODO:

- Fine-tune and test on sample of data for preliminary results.

## Feature Selection

To reduce the dimensionality and complexity of our models, we select the k-best features
by using a **mutual information** or **chi square** algorithm (from scikit-learn).

We experiment only on the best performing model from the previous experiments for both 4 classes and 2 classes classification. 
The goal is to see if we can reach optimal performance while using less features, therefore optimizing the model.

## Results and analysis

We perform test on the different test sets, using our top models for 4 classes and 2 classes, with the best
set of features and the optimal number of features.


## Conlusion and Future Work

----

