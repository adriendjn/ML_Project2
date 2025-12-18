import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(".")))
from utils import load_tweets, load_vocab, tweets_to_features, create_csv_submission
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from constants import SEED, EMBEDDING_FILE
import random

random.seed(SEED)
# Loading the data
tweet_pos, tweet_neg, tweet_test = load_tweets()

# Cleaning the index from every test tweets.
embedding_vocab = load_vocab()
embedding = np.load(EMBEDDING_FILE)

# Transforming the tweets using our embedding and vocab to construct features
list_vect_tweet_pos = tweets_to_features(tweet_pos, embedding_vocab, embedding)
list_vect_tweet_neg = tweets_to_features(tweet_neg, embedding_vocab, embedding)
list_vect_tweet_test = tweets_to_features(tweet_test, embedding_vocab, embedding)
# Constructing an array X_train and y_train combining the positive and negative vectors
X_train = np.vstack([list_vect_tweet_pos, list_vect_tweet_neg])
y_train = np.hstack([np.ones(len(list_vect_tweet_pos)), -np.ones(len(list_vect_tweet_neg))])
# Normalize train and test sets
X_train = normalize(X_train)
list_vect_tweet_test = normalize(list_vect_tweet_test)
# Logistic regresion + fitting of it
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
# Final prediction
y_pred = clf.predict(list_vect_tweet_test)
# Creating the ids array
ids = np.arange(1, len(y_pred)+1)
# Creating the submission file
create_csv_submission(ids, y_pred, "submission_glove.csv")