import numpy as np
from utils_glove import load_tweets, load_vocab, tweets_to_features, create_csv_submission
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
import random

random.seed(42)

TWITTER_DATASET_PATH = "../twitter-datasets/"

TRAIN_POS_FILE = TWITTER_DATASET_PATH + "train_pos.txt"
TRAIN_NEG_FILE = TWITTER_DATASET_PATH + "train_neg.txt"
TEST_DATA_FILE = TWITTER_DATASET_PATH + "test_data.txt"
EMBEDDING_VOCAB_FILE = "../vocab_cut.txt"
EMBEDDING_FILE = "../embeddings.npy"

tweet_pos, tweet_neg, tweet_test = load_tweets(TRAIN_POS_FILE, TRAIN_NEG_FILE, TEST_DATA_FILE)

# Cleaning the index from every test tweets.
embedding_vocab = load_vocab(EMBEDDING_VOCAB_FILE)
embedding = np.load(EMBEDDING_FILE)


list_vect_tweet_pos = tweets_to_features(tweet_pos, embedding_vocab, embedding)
list_vect_tweet_neg = tweets_to_features(tweet_neg, embedding_vocab, embedding)
list_vect_tweet_test = tweets_to_features(tweet_test, embedding_vocab, embedding)

X_train = np.vstack([list_vect_tweet_pos, list_vect_tweet_neg])
y_train = np.hstack([np.ones(len(list_vect_tweet_pos)), -np.ones(len(list_vect_tweet_neg))])

X_train = normalize(X_train)
list_vect_tweet_test = normalize(list_vect_tweet_test)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(list_vect_tweet_test)
ids = np.arange(1, len(y_pred)+1)
create_csv_submission(ids, y_pred, "../submissions/submissionGloveOnly.csv")