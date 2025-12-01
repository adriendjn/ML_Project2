import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter
from utils import load_tweets, load_vocab, tweets_to_features, create_csv_submission
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

train_pos = "twitter-datasets/train_pos.txt"
train_neg = "twitter-datasets/train_neg.txt"
final_test = "twitter-datasets/test_data.txt"
embedding_vocab_file = "vocab_cut.txt"

tweets_train_pos, tweets_train_neg, tweets_test = load_tweets(train_pos, train_neg, final_test)

# Cleaning the index from every test tweets.
embedding = np.load("embeddings.npy")
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
print(len(ids))
create_csv_submission(ids, y_pred, "submission.csv")

# --- Évaluation rapide sur un split train/test (peut être supprimé facilement) ---
if __name__ == "__main__":
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import accuracy_score, f1_score
	X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
	clf_val = LogisticRegression()
	clf_val.fit(X_tr, y_tr)
	y_val_pred = clf_val.predict(X_val)
	acc = accuracy_score(y_val, y_val_pred)
	f1 = f1_score(y_val, y_val_pred)
	print(f"[SPLIT EVAL] Accuracy: {acc:.4f} | F1: {f1:.4f}")