import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from utils import load_tweets, load_vocab, build_vocabulary, tweets_to_features
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from os import path


tweet_pos, tweet_neg, data_test = load_tweets("twitter-datasets/train_pos.txt","twitter-datasets/train_neg.txt", "twitter-datasets/test_data.txt")
vocab_file = load_vocab("vocab_cut.txt")
vocabulary_pos, word_counts_pos = build_vocabulary(tweet_pos, min_freq=5)
vocabulary_neg, word_counts_neg = build_vocabulary(tweet_neg, min_freq=5)
fp = path.join(path.split(__file__)[0], path.normcase("embeddings.npy"))
embeddings = np.load(fp)
list_vect_tweet_pos = tweets_to_features(tweet_pos, vocab_file, embeddings)
list_vect_tweet_neg = tweets_to_features(tweet_neg, vocab_file, embeddings)
list_vect_tweet_test = tweets_to_features(data_test, vocab_file, embeddings)


X_train = np.vstack([list_vect_tweet_pos, list_vect_tweet_neg])
y_train = np.hstack([ np.ones(len(list_vect_tweet_pos)),   np.zeros(len(list_vect_tweet_neg))])
    
X_train_full = X_train
y_train_full = y_train

X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)



clf = LogisticRegression(max_iter=2000, n_jobs=-1)
clf.fit(X_train_split, y_train_split)
print("Training Model finish")
y_val_pred = clf.predict(X_val)
acc = accuracy_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)

print("Validation Accuracy:", acc)
print("Validation F1-score:", f1)
