import numpy as np
from utils import load_tweets, load_vocab, tweets_to_features, create_csv_submission
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
import random

train_pos = "../twitter-datasets/train_pos.txt"
train_neg = "../twitter-datasets/train_neg.txt"
final_test = "../twitter-datasets/test_data.txt"
embedding_vocab_file = "../vocab_cut.txt"
embedding_file = "../embeddings.npy"

tweets_train_pos, tweets_train_neg, tweets_test = load_tweets(train_pos, train_neg, final_test)

random.seed(42)

# Cleaning the index from every test tweets.
embedding = np.load(embedding_file)
embedding_vocab = load_vocab(embedding_vocab_file)


X_pos = tweets_to_features(tweets_train_pos, embedding_vocab, embedding)
X_neg = tweets_to_features(tweets_train_neg, embedding_vocab, embedding)
X_test = tweets_to_features(tweets_test, embedding_vocab, embedding)

X_train = np.vstack([X_pos, X_neg])
y_train = np.hstack([np.ones(len(X_pos)), -np.ones(len(X_neg))])

X_train = normalize(X_train)
X_test = normalize(X_test)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
ids = np.arange(1, len(y_pred)+1)
create_csv_submission(ids, y_pred, "submissionGloveOnly.csv")