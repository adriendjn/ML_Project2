import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from utils import load_tweets, load_vocab, build_vocabulary, tweets_to_features, create_csv_submission
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack
import random

random.seed(42)

train_pos = "../twitter-datasets/train_pos.txt"
train_neg = "../twitter-datasets/train_neg.txt"
final_test = "../twitter-datasets/test_data.txt"
embedding_vocab_file = "../vocab_cut.txt"
embedding_file = "../embeddings.npy"


tweet_pos, tweet_neg, data_test = load_tweets(train_pos, train_neg, final_test)
vocab_file = load_vocab("../vocab_cut.txt")
data = np.load("../embeddings.npy")


vocabulary_pos, word_counts_pos = build_vocabulary(tweet_pos, min_freq=5)
vocabulary_neg, word_counts_neg = build_vocabulary(tweet_neg, min_freq=5)


list_vect_tweet_pos = tweets_to_features(tweet_pos, vocab_file, data)
list_vect_tweet_neg = tweets_to_features(tweet_neg, vocab_file, data)
list_vect_tweet_test = tweets_to_features(data_test, vocab_file, data)


X_embeddings = np.concatenate([list_vect_tweet_pos, list_vect_tweet_neg])
scaler = MinMaxScaler()
X_embeddings_scaled = scaler.fit_transform(X_embeddings)


tweets = tweet_pos + tweet_neg
y = np.array([1]*len(tweet_pos) + [-1]*len(tweet_neg))


vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
X_tfidf = vectorizer.fit_transform(tweets)
X_final = hstack([X_tfidf, X_embeddings_scaled])
X_train, X_val, y_train_split, y_val = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)



clf = LinearSVC()
clf.fit(X_train, y_train_split)
y_val_pred = clf.predict(X_val)

acc = accuracy_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)


print("Validation Accuracy:", acc)
print("Validation F1-score:", f1)

# Creating the submission

X_test_embeddings_scaled = scaler.transform(list_vect_tweet_test)
X_test_tfidf = vectorizer.transform(data_test)
X_test = hstack([X_test_tfidf, X_test_embeddings_scaled])
y_pred = clf.predict(X_test)
ids = np.arange(1, len(y_pred)+1)
create_csv_submission(ids, y_pred, "submissiontfidfglove.csv") 