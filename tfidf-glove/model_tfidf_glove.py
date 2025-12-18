import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import sys
import os
sys.path.append(os.path.abspath(os.path.join(".")))
from utils import load_tweets, load_vocab, tweets_to_features, create_csv_submission
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack
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

# Creating the normalized embeddings array
X_embeddings = np.concatenate([list_vect_tweet_pos, list_vect_tweet_neg])
scaler = MinMaxScaler()
X_embeddings_scaled = scaler.fit_transform(X_embeddings)

# Creating the tweet array for tfidf
tweets = tweet_pos + tweet_neg
y = np.array([1]*len(tweet_pos) + [-1]*len(tweet_neg))

# Making the tfidf array
vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
X_tfidf = vectorizer.fit_transform(tweets)
# Making the combined X of tfidf and glove
X_final = hstack([X_tfidf, X_embeddings_scaled])
# Splitting the data for split validation
X_train, X_val, y_train_split, y_val = train_test_split(
    X_final, y, test_size=0.2, random_state=SEED, stratify=y
)

# Using linearSVC to make prediction
clf = LinearSVC()
clf.fit(X_train, y_train_split)
y_val_pred = clf.predict(X_val)
# Computing the metrics on validation set
acc = accuracy_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)

print("Validation Accuracy:", acc)
print("Validation F1-score:", f1)

# Creating the submission
X_test_embeddings_scaled = scaler.transform(list_vect_tweet_test)
X_test_tfidf = vectorizer.transform(tweet_test)
X_test = hstack([X_test_tfidf, X_test_embeddings_scaled])
y_pred = clf.predict(X_test)
ids = np.arange(1, len(y_pred)+1)
create_csv_submission(ids, y_pred, "submission_tfidf-glove.csv") 